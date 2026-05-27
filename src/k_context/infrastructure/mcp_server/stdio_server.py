"""本地知识库工具的真实 stdio MCP server 入口。"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from k_context.application.mcp_contracts import SEARCH_KNOWLEDGE_BASE_TOOL
from k_context.application.mcp_registry import (
    MCPToolRegistry,
    create_default_mcp_tool_registry,
)
from k_context.infrastructure.storage.local_store import LocalKnowledgeBaseStore


SERVER_NAME = "k-context-local-knowledge-base"
VECTOR_WORKER_TIMEOUT_SECONDS_ENV = "KCONTEXT_MCP_VECTOR_WORKER_TIMEOUT_SECONDS"
DEFAULT_VECTOR_WORKER_TIMEOUT_SECONDS = 300.0

_VECTOR_WORKER_CODE = r"""
import contextlib
import json
import sys
from pathlib import Path

from k_context.application.mcp_contracts import SEARCH_KNOWLEDGE_BASE_TOOL
from k_context.application.mcp_registry import create_default_mcp_tool_registry

payload = json.load(sys.stdin)
root = Path(payload["root"])
arguments = payload["arguments"]
with contextlib.redirect_stdout(sys.stderr):
    result = create_default_mcp_tool_registry(root).call_tool(
        SEARCH_KNOWLEDGE_BASE_TOOL,
        arguments,
    )
sys.stdout.write(json.dumps(result, ensure_ascii=True))
"""


def create_stdio_mcp_server(
    *,
    root: Path,
    registry: MCPToolRegistry | None = None,
    validate_root: bool = True,
) -> FastMCP:
    """创建绑定到本地知识库 root 的 FastMCP server。"""

    bound_root = root.expanduser().resolve()
    if validate_root:
        LocalKnowledgeBaseStore().require_initialized(bound_root)
    tool_registry = registry or create_default_mcp_tool_registry(bound_root)
    server = FastMCP(name=SERVER_NAME)

    async def search_knowledge_base(
        query: str,
        top_k: int | None = None,
        retrieval_mode: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """搜索 MCP server 启动时绑定的知识库。"""

        # MCP stdio 将 stdout 保留给 JSON-RPC。部分 embedding/vector-store
        # 库会在向量检索时向 stdout 写入进度文本，因此工具执行期间
        # 将工具内部 stdout 重定向到 stderr。
        with redirect_stdout(sys.stderr):
            arguments = {
                "query": query,
                "top_k": top_k,
                "retrieval_mode": retrieval_mode,
                "filters": filters or {},
            }
            if registry is None and str(retrieval_mode or "").casefold() == "vector":
                return _call_default_tool_in_vector_worker(bound_root, arguments)
            return tool_registry.call_tool(SEARCH_KNOWLEDGE_BASE_TOOL, arguments)

    schema = next(
        tool for tool in tool_registry.list_tools() if tool["name"] == SEARCH_KNOWLEDGE_BASE_TOOL
    )
    server.add_tool(
        search_knowledge_base,
        name=SEARCH_KNOWLEDGE_BASE_TOOL,
        description=str(schema["description"]),
    )
    return server


def run_stdio_mcp_server(root: Path) -> None:
    """通过 stdio 运行本地 MCP server。"""

    server = create_stdio_mcp_server(root=root)
    server.run(transport="stdio")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="k-context-mcp-server",
        description="Run the K-Context local knowledge-base MCP server over stdio.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Knowledge-base root to bind at server startup.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_stdio_mcp_server(args.root)
    return 0


def _call_default_tool_in_vector_worker(root: Path, arguments: dict[str, Any]) -> dict[str, Any]:
    """在 FastMCP 工具运行器之外的普通子进程中运行向量检索。

    sentence-transformers/Chroma 向量检索在普通 Python 进程中较可靠，
    但在 MCP server 的工具运行器内部执行时可能卡住。server 仍是真实的
    stdio MCP 边界；该 worker 只隔离注册工具背后的重型向量检索实现。
    """

    timeout_seconds = _vector_worker_timeout_seconds()
    payload = json.dumps(
        {"root": str(root), "arguments": arguments},
        ensure_ascii=False,
    )
    env = os.environ.copy()
    env.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    env.setdefault("TQDM_DISABLE", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        completed = subprocess.run(
            [sys.executable, "-c", _VECTOR_WORKER_CODE],
            input=payload,
            text=True,
            encoding="utf-8",
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(
            f"vector retrieval worker timed out after {timeout_seconds:g} seconds"
        ) from exc
    if completed.returncode != 0:
        raise RuntimeError(
            f"vector retrieval worker failed with exit code {completed.returncode}"
        )
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("vector retrieval worker returned invalid JSON") from exc
    if not isinstance(result, dict):
        raise RuntimeError("vector retrieval worker result must be a JSON object")
    return result


def _vector_worker_timeout_seconds() -> float:
    raw_value = os.environ.get(VECTOR_WORKER_TIMEOUT_SECONDS_ENV, "").strip()
    if not raw_value:
        return DEFAULT_VECTOR_WORKER_TIMEOUT_SECONDS
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{VECTOR_WORKER_TIMEOUT_SECONDS_ENV} must be a number") from exc
    if value <= 0:
        raise ValueError(f"{VECTOR_WORKER_TIMEOUT_SECONDS_ENV} must be positive")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
