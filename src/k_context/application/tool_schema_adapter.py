"""从 MCP 工具 schema 到 OpenAI-compatible Chat Completions tools 的适配器。"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from k_context.application.mcp_client_bridge import MCPListedTool
from k_context.application.mcp_contracts import (
    MCPToolSchema,
    RESERVED_FILTER_KEYS,
    SEARCH_KNOWLEDGE_BASE_TOOL,
    SUPPORTED_TOOL_RETRIEVAL_MODES,
    search_knowledge_base_tool_schema,
)


class ToolSchemaAdapterError(ValueError):
    """当 MCP 工具 schema 无法安全适配时抛出。"""


@dataclass(frozen=True)
class OpenAIToolsSchemaAdapterResult:
    """转换后的 OpenAI-compatible 工具 schema 及回调映射。"""

    tools: tuple[dict[str, Any], ...]
    tool_name_mapping: Mapping[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "tools": [copy.deepcopy(tool) for tool in self.tools],
            "tool_name_mapping": dict(self.tool_name_mapping),
        }


def mcp_tools_to_openai_tools(
    tools: Iterable[MCPListedTool | MCPToolSchema | Mapping[str, Any]],
) -> OpenAIToolsSchemaAdapterResult:
    """将 MCP 工具转换为 OpenAI-compatible Chat Completions tools。

    返回的 schema 适用于 OpenAI-compatible chat-completions provider 使用的
    `tools` 请求参数。它有意省略已绑定的知识库路径：
    MCP server 在启动时负责该绑定。
    """

    converted: list[dict[str, Any]] = []
    mapping: dict[str, str] = {}
    for raw_tool in tools:
        normalized = _normalize_tool(raw_tool)
        parameters = _openai_parameters(normalized.input_schema, normalized.name)
        function_name = normalized.name
        converted.append(
            {
                "type": "function",
                "function": {
                    "name": function_name,
                    "description": _safe_description(normalized.description),
                    "parameters": parameters,
                },
            }
        )
        mapping[function_name] = normalized.name
    if not converted:
        raise ToolSchemaAdapterError("No MCP tools were provided for OpenAI schema conversion.")
    return OpenAIToolsSchemaAdapterResult(tools=tuple(converted), tool_name_mapping=mapping)


def required_tool_choice(tool_name: str) -> dict[str, Any]:
    """返回某个工具对应的 OpenAI-compatible required function tool_choice。"""

    if not str(tool_name).strip():
        raise ToolSchemaAdapterError("tool_name must not be empty.")
    return {"type": "function", "function": {"name": str(tool_name)}}


def tool_result_to_openai_tool_message(tool_call_id: str, result: Mapping[str, Any]) -> dict[str, str]:
    """将一个 MCP 工具结果载荷转换为 OpenAI tool 消息。"""

    if not str(tool_call_id).strip():
        raise ToolSchemaAdapterError("tool_call_id must not be empty.")
    return {
        "role": "tool",
        "tool_call_id": str(tool_call_id),
        "content": json.dumps(dict(result), ensure_ascii=False),
    }


@dataclass(frozen=True)
class _NormalizedTool:
    name: str
    description: str
    input_schema: Mapping[str, Any]


def _normalize_tool(raw_tool: MCPListedTool | MCPToolSchema | Mapping[str, Any]) -> _NormalizedTool:
    if isinstance(raw_tool, MCPListedTool):
        name = raw_tool.name
        description = raw_tool.description
        input_schema = raw_tool.input_schema
    elif isinstance(raw_tool, MCPToolSchema):
        name = raw_tool.name
        description = raw_tool.description
        input_schema = raw_tool.input_schema
    elif isinstance(raw_tool, Mapping):
        name = raw_tool.get("name")
        description = raw_tool.get("description")
        input_schema = raw_tool.get("inputSchema", raw_tool.get("input_schema"))
    else:
        raise ToolSchemaAdapterError(f"Unsupported MCP tool schema object: {type(raw_tool).__name__}.")

    if not isinstance(name, str) or not name.strip():
        raise ToolSchemaAdapterError("MCP tool schema is missing a non-empty name.")
    if not isinstance(description, str) or not description.strip():
        raise ToolSchemaAdapterError(f"MCP tool schema {name!r} is missing a description.")
    if not isinstance(input_schema, Mapping):
        raise ToolSchemaAdapterError(f"MCP tool schema {name!r} is missing inputSchema.")
    if _contains_reserved_key(input_schema):
        raise ToolSchemaAdapterError(f"MCP tool schema {name!r} exposes reserved binding fields.")
    return _NormalizedTool(name=name.strip(), description=description.strip(), input_schema=input_schema)


def _openai_parameters(input_schema: Mapping[str, Any], tool_name: str) -> dict[str, Any]:
    parameters = copy.deepcopy(dict(input_schema))
    if parameters.get("type") != "object":
        raise ToolSchemaAdapterError(f"MCP tool schema {tool_name!r} must use object parameters.")
    properties = parameters.get("properties")
    if not isinstance(properties, Mapping):
        raise ToolSchemaAdapterError(f"MCP tool schema {tool_name!r} must define properties.")
    if tool_name == SEARCH_KNOWLEDGE_BASE_TOOL:
        _validate_search_knowledge_base_parameters(properties, parameters.get("required"))
        parameters = copy.deepcopy(dict(search_knowledge_base_tool_schema().input_schema))
    # 某些 OpenAI-compatible provider 对 JSON Schema 方言细节比 OpenAI 更严格。
    # 工具契约仍由 MCP server 强制校验，因此从面向 provider 的工具定义中省略该可选关键字。
    parameters.pop("additionalProperties", None)
    return _sanitize_schema_text(parameters)


def _validate_search_knowledge_base_parameters(
    properties: Mapping[str, Any],
    required: Any,
) -> None:
    expected = {"query", "top_k", "retrieval_mode", "filters"}
    missing = expected.difference(str(key) for key in properties)
    if missing:
        raise ToolSchemaAdapterError(
            "search_knowledge_base schema is missing required properties: "
            f"{', '.join(sorted(missing))}."
        )
    if not isinstance(required, list | tuple) or "query" not in required:
        raise ToolSchemaAdapterError("search_knowledge_base schema must require query.")
    retrieval_mode = properties.get("retrieval_mode")
    if not isinstance(retrieval_mode, Mapping):
        raise ToolSchemaAdapterError("search_knowledge_base retrieval_mode schema is invalid.")
    enum_values = retrieval_mode.get("enum")
    if enum_values is not None and set(enum_values) != SUPPORTED_TOOL_RETRIEVAL_MODES:
        raise ToolSchemaAdapterError("search_knowledge_base retrieval_mode enum must be keyword/vector.")
    filters = properties.get("filters")
    if not isinstance(filters, Mapping):
        raise ToolSchemaAdapterError("search_knowledge_base filters schema is invalid.")
    if _contains_reserved_key(filters):
        raise ToolSchemaAdapterError("search_knowledge_base filters schema exposes reserved fields.")


def _sanitize_schema_text(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _sanitize_schema_text(nested) for key, nested in value.items()}
    if isinstance(value, list):
        return [_sanitize_schema_text(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_schema_text(item) for item in value]
    if isinstance(value, str):
        return _safe_description(value)
    return value


def _safe_description(value: str) -> str:
    sanitized = str(value)
    for key in RESERVED_FILTER_KEYS:
        sanitized = re.sub(rf"\b{re.escape(key)}\b", "bound knowledge-base path", sanitized, flags=re.IGNORECASE)
    return sanitized


def _contains_reserved_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key).casefold() in RESERVED_FILTER_KEYS:
                return True
            if _contains_reserved_key(nested):
                return True
    elif isinstance(value, list | tuple):
        return any(_contains_reserved_key(item) for item in value)
    return False
