"""基础设施 stdio MCP client bridge。"""

from __future__ import annotations

import json
import os
import queue
import sys
import threading
from contextlib import AsyncExitStack
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Mapping

import anyio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from k_context.application.mcp_client_bridge import (
    MCPClientBridgeError,
    MCPClientBridgeTimeoutError,
    MCPClientBridgeUnavailableError,
    MCPListedTool,
    validate_tool_arguments_do_not_override_root,
)


class StdioMCPClientBridge:
    """本地 stdio MCP server 子进程的同步 bridge。

    bridge 使用当前 Python 解释器启动 server，并且只在子进程
    启动参数中绑定知识库 root。工具调用参数会被校验，防止调用方
    覆盖该绑定 root。

    MCP SDK 的 stdio/session 上下文管理器是任务局部的，因此本类将其
    保持在一个长生命周期后台异步任务中，并通过线程安全队列通信。
    """

    def __init__(
        self,
        *,
        root: Path,
        timeout_seconds: float = 300.0,
        command: str | None = None,
        args: list[str] | None = None,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")
        self.root = root.expanduser().resolve()
        self.timeout_seconds = float(timeout_seconds)
        self.command = command or sys.executable
        self.args = list(args) if args is not None else [
            "-m",
            "k_context.infrastructure.mcp_server.stdio_server",
            "--root",
            str(self.root),
        ]
        self.cwd = cwd
        self.env = dict(env) if env is not None else os.environ.copy()
        # stdio MCP 将 stdout 保留给 JSON-RPC。禁用子进程中 embedding/model
        # 库的进度条，避免长时间向量调用污染或阻塞 stdio 协议流。
        self.env.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        self.env.setdefault("TQDM_DISABLE", "1")
        self.env.setdefault("TOKENIZERS_PARALLELISM", "false")
        self._ready_queue: queue.Queue[BaseException | None] = queue.Queue(maxsize=1)
        self._request_queue: queue.Queue[_BridgeRequest | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._initialized = False
        self._closed = False
        self._lock = threading.RLock()

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def is_closed(self) -> bool:
        return self._closed

    def initialize(self) -> None:
        """启动 server 子进程并执行一次 MCP initialize。"""

        with self._lock:
            if self._initialized:
                return
            if self._closed:
                raise MCPClientBridgeUnavailableError("MCP client bridge is closed.")
            self._thread = threading.Thread(
                target=self._run_worker_thread,
                name="k-context-mcp-stdio-client",
                daemon=True,
            )
            self._thread.start()
            try:
                startup_result = self._ready_queue.get(timeout=self.timeout_seconds)
            except queue.Empty as exc:
                self.close()
                raise MCPClientBridgeTimeoutError(
                    "Timed out while initializing MCP stdio server."
                ) from exc
            if startup_result is not None:
                self.close()
                raise MCPClientBridgeUnavailableError(
                    f"Failed to initialize MCP stdio server: {_safe_error_message(startup_result)}"
                ) from startup_result
            self._initialized = True

    def list_tools(self) -> tuple[MCPListedTool, ...]:
        """返回从 MCP server 发现的工具。"""

        result = self._request("list_tools")
        return tuple(_listed_tool(tool) for tool in result.tools)

    def call_tool(self, name: str, arguments: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """调用 MCP 工具并返回结构化字典载荷。"""

        safe_arguments = validate_tool_arguments_do_not_override_root(arguments)
        result = self._request("call_tool", name, safe_arguments)
        if getattr(result, "isError", False):
            raise MCPClientBridgeError(f"Failed to call MCP tool {name}: tool returned an error.")
        return _structured_content(result)

    def close(self) -> None:
        """关闭 MCP session 并终止子进程。"""

        thread: threading.Thread | None
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._initialized = False
            thread = self._thread
            if thread is not None and thread.is_alive():
                self._request_queue.put(None)
        if thread is not None and thread.is_alive():
            thread.join(timeout=self.timeout_seconds)

    def __enter__(self) -> "StdioMCPClientBridge":
        self.initialize()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def _request(self, method_name: str, *args: Any) -> Any:
        self.initialize()
        if self._closed:
            raise MCPClientBridgeUnavailableError("MCP client bridge is closed.")
        reply_queue: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)
        self._request_queue.put(_BridgeRequest(method_name=method_name, args=args, reply=reply_queue))
        try:
            success, payload = reply_queue.get(timeout=self.timeout_seconds)
        except queue.Empty as exc:
            self.close()
            raise MCPClientBridgeTimeoutError(f"Timed out during MCP operation: {method_name}.") from exc
        if success:
            return payload
        if isinstance(payload, TimeoutError):
            raise MCPClientBridgeTimeoutError(
                f"Timed out during MCP operation: {method_name}."
            ) from payload
        raise MCPClientBridgeError(
            f"Failed during MCP operation {method_name}: {_safe_error_message(payload)}"
        ) from payload

    def _run_worker_thread(self) -> None:
        try:
            anyio.run(self._worker_main)
        except BaseException as exc:  # pragma: no cover - defensive last-resort path
            _put_once(self._ready_queue, exc)

    async def _worker_main(self) -> None:
        try:
            async with AsyncExitStack() as exit_stack:
                parameters = StdioServerParameters(
                    command=self.command,
                    args=list(self.args),
                    env=dict(self.env),
                    cwd=self.cwd,
                )
                read_stream, write_stream = await exit_stack.enter_async_context(stdio_client(parameters))
                session = await exit_stack.enter_async_context(
                    ClientSession(
                        read_stream,
                        write_stream,
                        read_timeout_seconds=timedelta(seconds=self.timeout_seconds),
                    )
                )
                await session.initialize()
                self._ready_queue.put(None)
                await self._serve_requests(session)
        except BaseException as exc:
            _put_once(self._ready_queue, exc)

    async def _serve_requests(self, session: ClientSession) -> None:
        while True:
            request = await anyio.to_thread.run_sync(self._request_queue.get)
            if request is None:
                return
            try:
                if request.method_name == "list_tools":
                    payload = await session.list_tools()
                elif request.method_name == "call_tool":
                    payload = await session.call_tool(*request.args)
                else:
                    raise MCPClientBridgeError(f"Unsupported MCP bridge operation: {request.method_name}.")
            except BaseException as exc:
                request.reply.put((False, exc))
            else:
                request.reply.put((True, payload))


@dataclass(frozen=True)
class _BridgeRequest:
    method_name: str
    args: tuple[Any, ...]
    reply: queue.Queue[tuple[bool, Any]]


def _put_once(target: queue.Queue[BaseException | None], value: BaseException) -> None:
    try:
        target.put_nowait(value)
    except queue.Full:
        return


def _listed_tool(tool: Any) -> MCPListedTool:
    input_schema = getattr(tool, "inputSchema", None) or getattr(tool, "input_schema", None) or {}
    if not isinstance(input_schema, Mapping):
        input_schema = {}
    return MCPListedTool(
        name=str(getattr(tool, "name")),
        description=getattr(tool, "description", None),
        input_schema=input_schema,
    )


def _structured_content(result: Any) -> dict[str, Any]:
    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, Mapping):
        return dict(structured)
    content = getattr(result, "content", None) or []
    for block in content:
        text = getattr(block, "text", None)
        if not text:
            continue
        parsed = json.loads(str(text))
        if isinstance(parsed, Mapping):
            return dict(parsed)
    raise MCPClientBridgeError("MCP tool result did not contain structured dictionary content.")


def _safe_error_message(exc: BaseException) -> str:
    text = str(exc).strip() or exc.__class__.__name__
    for marker in ("Authorization", "Bearer", "KCONTEXT_LLM_API_KEY"):
        text = text.replace(marker, "[REDACTED]")
    return text
