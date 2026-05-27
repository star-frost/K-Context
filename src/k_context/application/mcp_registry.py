"""MCP 工具的应用层注册表。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

from k_context.application.mcp_contracts import (
    SEARCH_KNOWLEDGE_BASE_TOOL,
    MCPToolSchema,
    SearchKnowledgeBaseInput,
    search_knowledge_base_tool_schema,
)
from k_context.application.mcp_tools import SearchKnowledgeBaseTool


class MCPToolRegistryError(RuntimeError):
    """当已注册 MCP 工具无法列出或调用时抛出。"""


class MCPTool(Protocol):
    """注册在 MCP 后面的可调用应用工具。"""

    def call(self, tool_input: SearchKnowledgeBaseInput | Mapping[str, Any]) -> object:
        """使用已验证参数执行工具。"""


@dataclass(frozen=True)
class RegisteredMCPTool:
    """一个已注册的 MCP 工具绑定。"""

    schema: MCPToolSchema
    tool: MCPTool


class MCPToolRegistry:
    """本地 MCP server 暴露的应用工具注册表。"""

    def __init__(
        self,
        *,
        root: Path,
        search_tool: MCPTool | None = None,
    ) -> None:
        self.root = root.expanduser().resolve()
        self._tools: dict[str, RegisteredMCPTool] = {}
        self.register(
            schema=search_knowledge_base_tool_schema(),
            tool=search_tool or SearchKnowledgeBaseTool(root=self.root),
        )

    def register(self, *, schema: MCPToolSchema, tool: MCPTool) -> None:
        if schema.name in self._tools:
            raise MCPToolRegistryError(f"MCP tool is already registered: {schema.name}.")
        self._tools[schema.name] = RegisteredMCPTool(schema=schema, tool=tool)

    def list_tools(self) -> tuple[dict[str, Any], ...]:
        """以可序列化字典形式返回已注册工具 schema。"""

        return tuple(binding.schema.to_dict() for binding in self._tools.values())

    def call_tool(self, name: str, arguments: Mapping[str, Any] | None) -> dict[str, Any]:
        """调用已注册工具并返回可序列化结果载荷。"""

        try:
            binding = self._tools[name]
        except KeyError as exc:
            raise MCPToolRegistryError(f"Unknown MCP tool: {name}.") from exc

        try:
            result = binding.tool.call(dict(arguments or {}))
        except Exception as exc:
            raise MCPToolRegistryError(f"MCP tool call failed for {name}: {exc}") from exc

        to_dict = getattr(result, "to_dict", None)
        if callable(to_dict):
            converted = to_dict()
        elif isinstance(result, Mapping):
            converted = dict(result)
        else:
            raise MCPToolRegistryError(
                f"MCP tool {name} returned a non-serializable result: {type(result).__name__}."
            )
        if not isinstance(converted, dict):
            raise MCPToolRegistryError(f"MCP tool {name} result must serialize to a dict.")
        return converted


def create_default_mcp_tool_registry(root: Path) -> MCPToolRegistry:
    """创建绑定到知识库 root 的默认注册表。"""

    return MCPToolRegistry(root=root)
