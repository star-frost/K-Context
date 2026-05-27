"""MCP client bridge 实现的应用边界。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

from k_context.application.mcp_contracts import RESERVED_FILTER_KEYS


class MCPClientBridgeError(RuntimeError):
    """当 MCP client bridge 操作失败时抛出。"""


class MCPClientBridgeUnavailableError(MCPClientBridgeError):
    """当本地 MCP server 无法启动或无法访问时抛出。"""


class MCPClientBridgeTimeoutError(MCPClientBridgeUnavailableError):
    """当 MCP bridge 操作超过配置超时时抛出。"""


@dataclass(frozen=True)
class MCPListedTool:
    """从 MCP server 发现的工具的可序列化摘要。"""

    name: str
    description: str | None = None
    input_schema: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": dict(self.input_schema),
        }


class MCPClientBridge(Protocol):
    """与本地 stdio MCP server 通信的客户端协议。"""

    def initialize(self) -> None:
        """启动或连接 MCP server 并执行 initialize。"""

    def list_tools(self) -> tuple[MCPListedTool, ...]:
        """返回 MCP server 公布的工具。"""

    def call_tool(self, name: str, arguments: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """按名称调用工具并返回结构化载荷。"""

    def close(self) -> None:
        """关闭客户端并终止本地 MCP server 子进程。"""


def validate_tool_arguments_do_not_override_root(
    arguments: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """拒绝 root 覆盖尝试后返回工具参数副本。"""

    copied = dict(arguments or {})
    forbidden = _find_reserved_root_path(copied)
    if forbidden is not None:
        raise MCPClientBridgeError(
            "MCP tool arguments must not include knowledge-base binding fields: "
            f"{forbidden}."
        )
    return copied


def _find_reserved_root_path(value: Any, *, prefix: str = "arguments") -> str | None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key)
            path = f"{prefix}.{key_text}"
            if key_text in RESERVED_FILTER_KEYS:
                return path
            found = _find_reserved_root_path(nested, prefix=path)
            if found is not None:
                return found
    elif isinstance(value, list | tuple):
        for index, nested in enumerate(value):
            found = _find_reserved_root_path(nested, prefix=f"{prefix}[{index}]")
            if found is not None:
                return found
    return None
