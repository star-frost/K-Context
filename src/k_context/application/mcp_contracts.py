"""MCP 知识库工具的应用层契约。

本模块有意只包含可序列化契约对象。它不启动 MCP server，
不调用 tools/list 或 tools/call，也不修改 ask 流程；这些职责由
后续 MCP 实现任务引入。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


SEARCH_KNOWLEDGE_BASE_TOOL = "search_knowledge_base"
MCP_SERVER_TRANSPORT_STDIO = "stdio"
MCP_EVENT_SERVER_START = "mcp_server_start"
MCP_EVENT_TOOLS_LIST = "mcp_tools_list"
MCP_EVENT_TOOL_CALL = "mcp_tool_call"
MCP_EVENT_TOOL_LATENCY = "tool_latency"
MCP_EVENT_LLM_TOOL_LOOP_LATENCY = "llm_tool_loop_latency"
MCP_METRICS_EVENT_TYPES = {
    MCP_EVENT_SERVER_START,
    MCP_EVENT_TOOLS_LIST,
    MCP_EVENT_TOOL_CALL,
    MCP_EVENT_TOOL_LATENCY,
    MCP_EVENT_LLM_TOOL_LOOP_LATENCY,
}
SUPPORTED_TOOL_RETRIEVAL_MODES = {"vector", "keyword"}
RESERVED_FILTER_KEYS = {"root"}


@dataclass(frozen=True)
class SearchKnowledgeBaseInput:
    """search_knowledge_base MCP 工具的输入。

    知识库 root 有意不属于该契约。
    MCP server 会在进程启动时绑定 root。
    """

    query: str
    top_k: int | None = None
    retrieval_mode: str | None = None
    filters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty("query", self.query)
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k must be positive when provided.")
        if self.retrieval_mode is not None:
            mode = str(self.retrieval_mode).strip().casefold()
            if mode not in SUPPORTED_TOOL_RETRIEVAL_MODES:
                raise ValueError(
                    "Unsupported retrieval_mode: "
                    f"{self.retrieval_mode}. Supported values: "
                    f"{', '.join(sorted(SUPPORTED_TOOL_RETRIEVAL_MODES))}."
                )
        if not isinstance(self.filters, Mapping):
            raise ValueError("filters must be a mapping when provided.")
        forbidden_keys = RESERVED_FILTER_KEYS.intersection(str(key) for key in self.filters)
        if forbidden_keys:
            raise ValueError(
                "filters must not include reserved knowledge-base binding fields: "
                f"{', '.join(sorted(forbidden_keys))}."
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SearchKnowledgeBaseInput":
        return cls(
            query=str(data.get("query", "")),
            top_k=_optional_int(data.get("top_k")),
            retrieval_mode=_optional_str(data.get("retrieval_mode")),
            filters=_optional_mapping(data.get("filters")),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"query": self.query, "filters": dict(self.filters)}
        if self.top_k is not None:
            result["top_k"] = self.top_k
        if self.retrieval_mode is not None:
            result["retrieval_mode"] = str(self.retrieval_mode).strip().casefold()
        return result


@dataclass(frozen=True)
class SearchKnowledgeBaseResultItem:
    """search_knowledge_base 返回的一个来源。"""

    chunk_id: str
    source_doc_id: str
    source_doc_name: str
    score: float
    retrieval_mode: str
    block_ids: tuple[str, ...]
    page_start: int | None = None
    page_end: int | None = None
    quote: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty("chunk_id", self.chunk_id)
        _require_non_empty("source_doc_id", self.source_doc_id)
        _require_non_empty("source_doc_name", self.source_doc_name)
        _require_non_empty("retrieval_mode", self.retrieval_mode)
        if self.retrieval_mode not in SUPPORTED_TOOL_RETRIEVAL_MODES:
            raise ValueError(
                "retrieval_mode must be one of: "
                f"{', '.join(sorted(SUPPORTED_TOOL_RETRIEVAL_MODES))}."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "source_doc_id": self.source_doc_id,
            "source_doc_name": self.source_doc_name,
            "score": float(self.score),
            "retrieval_mode": self.retrieval_mode,
            "block_ids": list(self.block_ids),
            "page_start": self.page_start,
            "page_end": self.page_end,
            "quote": self.quote,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class SearchKnowledgeBaseMetrics:
    """MCP 工具结果载荷中返回的指标。"""

    retrieval_time_ms: float | None = None
    query_embedding_time_ms: float | None = None
    result_count: int | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result = dict(self.extra)
        if self.retrieval_time_ms is not None:
            result["retrieval_time_ms"] = max(0.0, float(self.retrieval_time_ms))
        if self.query_embedding_time_ms is not None:
            result["query_embedding_time_ms"] = max(0.0, float(self.query_embedding_time_ms))
        if self.result_count is not None:
            result["result_count"] = max(0, int(self.result_count))
        return result


@dataclass(frozen=True)
class SearchKnowledgeBaseOutput:
    """search_knowledge_base MCP 工具返回的输出载荷。"""

    results: tuple[SearchKnowledgeBaseResultItem, ...] = ()
    fallback_used: bool = False
    fallback_reason: str | None = None
    metrics: SearchKnowledgeBaseMetrics = field(default_factory=SearchKnowledgeBaseMetrics)

    def to_dict(self) -> dict[str, Any]:
        return {
            "results": [item.to_dict() for item in self.results],
            "fallback_used": self.fallback_used,
            "fallback_reason": self.fallback_reason,
            "metrics": self.metrics.to_dict(),
        }


@dataclass(frozen=True)
class MCPToolSchema:
    """MCP 工具 schema 的内部表示。"""

    name: str
    description: str
    input_schema: Mapping[str, Any]

    def __post_init__(self) -> None:
        _require_non_empty("name", self.name)
        _require_non_empty("description", self.description)
        if _contains_reserved_root_key(self.input_schema):
            raise ValueError("MCP tool input schema must not expose root.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": dict(self.input_schema),
        }


@dataclass(frozen=True)
class MCPToolCall:
    """MCP 工具调用请求的内部表示。"""

    tool_call_id: str
    name: str
    arguments: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_call_id": self.tool_call_id,
            "name": self.name,
            "arguments": dict(self.arguments),
        }


@dataclass(frozen=True)
class MCPToolResult:
    """成功 MCP 工具调用结果的内部表示。"""

    tool_call_id: str
    name: str
    content: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_call_id": self.tool_call_id,
            "name": self.name,
            "content": dict(self.content),
        }


@dataclass(frozen=True)
class MCPToolError:
    """失败 MCP 工具调用的内部表示。"""

    tool_call_id: str
    name: str
    error_message: str
    error_type: str = "tool_error"

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_call_id": self.tool_call_id,
            "name": self.name,
            "error_type": self.error_type,
            "error_message": self.error_message,
        }


@dataclass(frozen=True)
class MCPSessionFields:
    """为 MCP 工具调用问答流程预留的可选 sessions.jsonl 字段。"""

    tool_calls: tuple[MCPToolCall, ...] = ()
    tool_results_summary: Mapping[str, Any] = field(default_factory=dict)
    mcp_server_transport: str = MCP_SERVER_TRANSPORT_STDIO
    tool_loop_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_calls": [call.to_dict() for call in self.tool_calls],
            "tool_results_summary": dict(self.tool_results_summary),
            "mcp_server_transport": self.mcp_server_transport,
            "tool_loop_count": max(0, int(self.tool_loop_count)),
        }


def search_knowledge_base_tool_schema() -> MCPToolSchema:
    """返回文档规定的 search_knowledge_base MCP 工具 schema。"""

    return MCPToolSchema(
        name=SEARCH_KNOWLEDGE_BASE_TOOL,
        description=(
            "Search the bound local knowledge base for chunks relevant to a user query. "
            "The server binds the knowledge-base root at startup; root is not a tool input."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "User question or retrieval query.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Maximum number of results to return.",
                },
                "retrieval_mode": {
                    "type": "string",
                    "enum": sorted(SUPPORTED_TOOL_RETRIEVAL_MODES),
                    "description": "Retrieval mode.",
                },
                "filters": {
                    "type": "object",
                    "description": "Reserved filters. Must not contain knowledge-base binding fields.",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    )


def _require_non_empty(field_name: str, value: str) -> str:
    checked = str(value).strip()
    if not checked:
        raise ValueError(f"{field_name} must not be empty.")
    return checked


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("integer values must not be booleans.")
    return int(value)


def _optional_str(value: Any) -> str | None:
    return str(value) if value is not None else None


def _optional_mapping(value: Any) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("Expected a mapping.")
    return value


def _contains_reserved_root_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key) in RESERVED_FILTER_KEYS:
                return True
            if _contains_reserved_root_key(nested):
                return True
    if isinstance(value, list | tuple):
        return any(_contains_reserved_root_key(item) for item in value)
    return False
