from __future__ import annotations

import re
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Mapping

from k_context.application.answer_service import (
    EVIDENCE_BASIC,
    EVIDENCE_INSUFFICIENT,
    EVIDENCE_SUFFICIENT,
    AnswerSource,
    GroundedAnswer,
)
from k_context.application.llm_client import (
    LLMClient,
    LLMMessage,
    LLMRequest,
    TOKEN_USAGE_UNAVAILABLE,
    TokenUsage,
)
from k_context.application.mcp_client_bridge import MCPClientBridge, MCPClientBridgeError
from k_context.application.mcp_contracts import (
    MCP_EVENT_LLM_TOOL_LOOP_LATENCY,
    MCP_EVENT_SERVER_START,
    MCP_EVENT_TOOL_CALL,
    MCP_EVENT_TOOL_LATENCY,
    MCP_EVENT_TOOLS_LIST,
    MCP_SERVER_TRANSPORT_STDIO,
    SEARCH_KNOWLEDGE_BASE_TOOL,
)
from k_context.application.tool_schema_adapter import (
    ToolSchemaAdapterError,
    mcp_tools_to_openai_tools,
    required_tool_choice,
    tool_result_to_openai_tool_message,
)

_SENSITIVE_PATTERNS = (
    re.compile(r"KCONTEXT_LLM_API_KEY\s*=\s*\S+", re.IGNORECASE),
    re.compile(r"\bKCONTEXT_LLM_API_KEY\b", re.IGNORECASE),
    re.compile(r"Authorization\s*:\s*Bearer\s+\S+", re.IGNORECASE),
    re.compile(r"Bearer\s+[A-Za-z0-9._~+/=-]{8,}", re.IGNORECASE),
    re.compile(r"(api[_-]?key|apikey|secret)\s*[:=]\s*\S+", re.IGNORECASE),
    re.compile(r"sk-[A-Za-z0-9._-]+", re.IGNORECASE),
)



@dataclass(frozen=True)
class AskMCPToolLoop:
    """
    用于 kb ask 的单轮 MCP 工具调用循环。
    """

    llm_client: LLMClient
    mcp_bridge: MCPClientBridge
    max_tool_rounds: int = 1

    def __post_init__(self) -> None:
        if self.max_tool_rounds != 1:
            raise ValueError("AskMCPToolLoop currently supports max_tool_rounds=1 only.")

    def run(
        self,
        *,
        question: str,
        top_k: int | None = None,
        retrieval_mode: str | None = None,
    ) -> GroundedAnswer:
        query = question.strip()
        if not query:
            raise AskMCPToolLoopError("question must not be empty.")
        requested_mode = retrieval_mode or "vector"
        loop_start = perf_counter()
        events: list[dict[str, Any]] = []
        tool_calls_summary: tuple[Mapping[str, object], ...] = ()
        tool_results_summary: Mapping[str, object] = _empty_tool_results_summary()
        try:
            initialize_start = perf_counter()
            self.mcp_bridge.initialize()
            events.append(
                _metric_event(
                    MCP_EVENT_SERVER_START,
                    "stdio_initialize",
                    "success",
                    initialize_start,
                    {"mcp_server_transport": MCP_SERVER_TRANSPORT_STDIO},
                )
            )
            list_start = perf_counter()
            listed_tools = self.mcp_bridge.list_tools()
            events.append(
                _metric_event(
                    MCP_EVENT_TOOLS_LIST,
                    "tools_list",
                    "success",
                    list_start,
                    {
                        "mcp_server_transport": MCP_SERVER_TRANSPORT_STDIO,
                        "tool_names": [tool.name for tool in listed_tools],
                    },
                )
            )
            adapter_result = mcp_tools_to_openai_tools(listed_tools)
            tool_choice = required_tool_choice(SEARCH_KNOWLEDGE_BASE_TOOL)
            first_request = LLMRequest(
                question=query,
                messages=_initial_messages(query),
                tools=adapter_result.tools,
                tool_choice=tool_choice,
            )
            first_llm_start = perf_counter()
            first_response = self.llm_client.generate(first_request)
            if not first_response.tool_calls:
                events.append(
                    _metric_event(
                        MCP_EVENT_LLM_TOOL_LOOP_LATENCY,
                        "single_round_tool_loop",
                        "failure",
                        loop_start,
                        {
                            "fallback_used": True,
                            "fallback_reason": "llm_tool_call_missing",
                            "first_llm_latency_ms": _elapsed_ms(first_llm_start),
                            "tool_loop_count": 0,
                        },
                    )
                )
                return _fallback_answer(
                    question=query,
                    retrieval_mode=requested_mode,
                    top_k=top_k,
                    reason="llm_tool_call_missing",
                    tool_calls_used=False,
                    tool_loop_count=0,
                    mcp_metrics=_mcp_metrics(events),
                )
            tool_call = first_response.tool_calls[0]
            tool_calls_summary = (
                _tool_call_summary(tool_call.tool_call_id, tool_call.name, tool_call.arguments, status="pending"),
            )
            mcp_tool_name = adapter_result.tool_name_mapping.get(tool_call.name)
            if mcp_tool_name != SEARCH_KNOWLEDGE_BASE_TOOL:
                tool_calls_summary = (
                    _tool_call_summary(
                        tool_call.tool_call_id,
                        tool_call.name,
                        tool_call.arguments,
                        status="failure",
                    ),
                )
                events.append(
                    _metric_event(
                        MCP_EVENT_LLM_TOOL_LOOP_LATENCY,
                        "single_round_tool_loop",
                        "failure",
                        loop_start,
                        {
                            "fallback_used": True,
                            "fallback_reason": f"unknown_tool_name:{tool_call.name}",
                            "tool_loop_count": 1,
                            "tool_name": tool_call.name,
                        },
                    )
                )
                return _fallback_answer(
                    question=query,
                    retrieval_mode=requested_mode,
                    top_k=top_k,
                    reason=f"unknown_tool_name:{tool_call.name}",
                    tool_calls_used=True,
                    tool_loop_count=1,
                    tool_name=tool_call.name,
                    tool_calls=tool_calls_summary,
                    tool_results_summary=tool_results_summary,
                    mcp_metrics=_mcp_metrics(events),
                )
            tool_arguments = _tool_arguments(
                tool_call.arguments,
                question=query,
                top_k=top_k,
                retrieval_mode=retrieval_mode,
            )
            tool_start = perf_counter()
            try:
                tool_result = self.mcp_bridge.call_tool(mcp_tool_name, tool_arguments)
            except Exception as exc:  # noqa: BLE001
                safe_error = _sanitize(str(exc)) or exc.__class__.__name__
                events.append(
                    _metric_event(
                        MCP_EVENT_TOOL_CALL,
                        mcp_tool_name,
                        "failure",
                        tool_start,
                        {
                            "tool_name": mcp_tool_name,
                            "arguments_summary": _arguments_summary(tool_arguments),
                            "error_message": safe_error,
                        },
                    )
                )
                events.append(
                    _metric_event(
                        MCP_EVENT_TOOL_LATENCY,
                        mcp_tool_name,
                        "failure",
                        tool_start,
                        {"tool_name": mcp_tool_name, "error_message": safe_error},
                    )
                )
                raise
            tool_duration_ms = _elapsed_ms(tool_start)
            events.append(
                _metric_event(
                    MCP_EVENT_TOOL_CALL,
                    mcp_tool_name,
                    "success",
                    tool_start,
                    {
                        "tool_name": mcp_tool_name,
                        "arguments_summary": _arguments_summary(tool_arguments),
                    },
                )
            )
            events.append(
                _metric_event(
                    MCP_EVENT_TOOL_LATENCY,
                    mcp_tool_name,
                    "success",
                    tool_start,
                    {"tool_name": mcp_tool_name, "result_count": _raw_result_count(tool_result)},
                    duration_ms=tool_duration_ms,
                )
            )
            tool_calls_summary = (
                _tool_call_summary(
                    tool_call.tool_call_id,
                    mcp_tool_name,
                    tool_arguments,
                    status="success",
                ),
            )
            tool_results_summary = _tool_results_summary(mcp_tool_name, tool_result)
            result_items = tuple(_result_item(item) for item in _result_rows(tool_result))
            if not result_items:
                events.append(
                    _metric_event(
                        MCP_EVENT_LLM_TOOL_LOOP_LATENCY,
                        "single_round_tool_loop",
                        "success",
                        loop_start,
                        {
                            "fallback_used": True,
                            "fallback_reason": "mcp_tool_result_empty",
                            "tool_loop_count": 1,
                            "tool_name": mcp_tool_name,
                        },
                    )
                )
                return _fallback_answer(
                    question=query,
                    retrieval_mode=requested_mode,
                    top_k=top_k,
                    reason="mcp_tool_result_empty",
                    tool_calls_used=True,
                    tool_loop_count=1,
                    tool_name=mcp_tool_name,
                    tool_calls=tool_calls_summary,
                    tool_results_summary=tool_results_summary,
                    mcp_metrics=_mcp_metrics(events),
                )
            tool_message_payload = tool_result_to_openai_tool_message(
                tool_call.tool_call_id,
                tool_result,
            )
            final_request = LLMRequest(
                question=query,
                messages=first_request.messages
                + (
                    LLMMessage(role="assistant", content="", tool_calls=(tool_call,)),
                    LLMMessage(
                        role="tool",
                        content=tool_message_payload["content"],
                        tool_call_id=tool_call.tool_call_id,
                        name=mcp_tool_name,
                    ),
                ),
                sources=tuple(_source_payload(source) for source in result_items),
            )
            final_llm_start = perf_counter()
            final_response = self.llm_client.generate(final_request)
            events.append(
                _metric_event(
                    MCP_EVENT_LLM_TOOL_LOOP_LATENCY,
                    "single_round_tool_loop",
                    "success",
                    loop_start,
                    {
                        "fallback_used": False,
                        "tool_loop_count": 1,
                        "tool_name": mcp_tool_name,
                        "first_llm_latency_ms": _elapsed_ms(first_llm_start),
                        "final_llm_latency_ms": _elapsed_ms(final_llm_start),
                        "tool_latency_ms": tool_duration_ms,
                    },
                )
            )
            return GroundedAnswer(
                answer=final_response.answer,
                evidence_level=EVIDENCE_SUFFICIENT if len(result_items) > 1 else EVIDENCE_BASIC,
                sources=result_items,
                retrieval_mode=_retrieval_mode(result_items, requested_mode),
                top_k=top_k,
                fallback_used=False,
                fallback_reason=None,
                token_usage=final_response.token_usage,
                token_usage_source=final_response.token_usage_source,
                latency_ms=final_response.latency_ms,
                tool_calls_used=True,
                mcp_server_transport=MCP_SERVER_TRANSPORT_STDIO,
                tool_name=mcp_tool_name,
                tool_loop_count=1,
                tool_calls=tool_calls_summary,
                tool_results_summary=tool_results_summary,
                mcp_metrics=_mcp_metrics(events),
            )
        except (MCPClientBridgeError, ToolSchemaAdapterError, Exception) as exc:  # noqa: BLE001
            safe_detail = _sanitize(str(exc)) or exc.__class__.__name__
            reason = f"mcp_tool_loop_failed:{safe_detail}"
            if tool_calls_summary and tool_calls_summary[0].get("status") == "pending":
                tool_calls_summary = (
                    {
                        **dict(tool_calls_summary[0]),
                        "status": "failure",
                    },
                )
            if _has_no_event(events, MCP_EVENT_LLM_TOOL_LOOP_LATENCY):
                events.append(
                    _metric_event(
                        MCP_EVENT_LLM_TOOL_LOOP_LATENCY,
                        "single_round_tool_loop",
                        "failure",
                        loop_start,
                        {
                            "fallback_used": True,
                            "fallback_reason": reason,
                            "tool_loop_count": 1 if tool_calls_summary else 0,
                            "error_message": safe_detail,
                        },
                    )
                )
            return _fallback_answer(
                question=query,
                retrieval_mode=requested_mode,
                top_k=top_k,
                reason=reason,
                tool_calls_used=False,
                tool_loop_count=1 if tool_calls_summary else 0,
                tool_calls=tool_calls_summary,
                tool_results_summary=tool_results_summary,
                mcp_metrics=_mcp_metrics(events),
            )
        finally:
            self.mcp_bridge.close()


def _initial_messages(question: str) -> tuple[LLMMessage, ...]:
    return (
        LLMMessage(
            role="system",
            content=(
                "You are K-Context. You must call the search_knowledge_base tool exactly once "
                "before answering. Do not answer from prior knowledge."
            ),
        ),
        LLMMessage(role="user", content=question),
    )


def _tool_arguments(
    arguments: Mapping[str, Any],
    *,
    question: str,
    top_k: int | None,
    retrieval_mode: str | None,
) -> dict[str, Any]:
    copied = dict(arguments)
    copied.setdefault("query", question)
    if top_k is not None:
        copied["top_k"] = top_k
    if retrieval_mode is not None:
        copied["retrieval_mode"] = retrieval_mode
    return copied


def _result_rows(tool_result: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    rows = tool_result.get("results", ())
    if not isinstance(rows, list | tuple):
        raise AskMCPToolLoopError("search_knowledge_base result.results must be a list.")
    return tuple(row for row in rows if isinstance(row, Mapping))


def _result_item(row: Mapping[str, Any]) -> AnswerSource:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), Mapping) else {}
    return AnswerSource(
        chunk_id=str(row.get("chunk_id", "")),
        source_doc_id=str(row.get("source_doc_id", "")),
        source_doc_name=str(row.get("source_doc_name", "")),
        score=float(row.get("score", 0.0)),
        block_ids=tuple(str(item) for item in row.get("block_ids", ()) or ()),
        retrieval_mode=str(row.get("retrieval_mode") or "vector"),
        page_start=_optional_int(row.get("page_start")),
        page_end=_optional_int(row.get("page_end")),
        quote=str(row.get("quote")) if row.get("quote") is not None else None,
        metadata=dict(metadata),
    )


def _source_payload(source: AnswerSource) -> Mapping[str, Any]:
    return {
        "chunk_id": source.chunk_id,
        "source_doc_id": source.source_doc_id,
        "source_doc_name": source.source_doc_name,
        "score": source.score,
        "retrieval_mode": source.retrieval_mode,
        "block_ids": list(source.block_ids),
        "page_start": source.page_start,
        "page_end": source.page_end,
        "quote": source.quote,
        "metadata": dict(source.metadata),
    }


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _retrieval_mode(sources: tuple[AnswerSource, ...], default: str) -> str:
    return sources[0].retrieval_mode if sources else default


def _fallback_answer(
    *,
    question: str,
    retrieval_mode: str,
    top_k: int | None,
    reason: str,
    tool_calls_used: bool,
    tool_loop_count: int,
    tool_name: str | None = None,
    tool_calls: tuple[Mapping[str, object], ...] = (),
    tool_results_summary: Mapping[str, object] | None = None,
    mcp_metrics: Mapping[str, object] | None = None,
) -> GroundedAnswer:
    del question
    return GroundedAnswer(
        answer="当前知识库依据不足，无法回答该问题。",
        evidence_level=EVIDENCE_INSUFFICIENT,
        sources=(),
        retrieval_mode=retrieval_mode,
        top_k=top_k,
        fallback_used=True,
        fallback_reason=reason,
        token_usage=TokenUsage.unavailable(),
        token_usage_source=TOKEN_USAGE_UNAVAILABLE,
        latency_ms=0.0,
        tool_calls_used=tool_calls_used,
        mcp_server_transport=MCP_SERVER_TRANSPORT_STDIO,
        tool_name=tool_name,
        tool_loop_count=tool_loop_count,
        tool_calls=tool_calls,
        tool_results_summary=tool_results_summary or _empty_tool_results_summary(),
        mcp_metrics=mcp_metrics or _mcp_metrics(()),
    )


def _sanitize(text: str) -> str:
    sanitized = str(text)
    for pattern in _SENSITIVE_PATTERNS:
        sanitized = pattern.sub("[REDACTED]", sanitized)
    return sanitized.strip()


def _elapsed_ms(start: float) -> float:
    return max(0.0, round((perf_counter() - start) * 1000, 3))


def _metric_event(
    event_type: str,
    operation: str,
    status: str,
    start: float,
    metadata: Mapping[str, Any] | None = None,
    *,
    duration_ms: float | None = None,
) -> dict[str, Any]:
    safe_metadata = _sanitize_mapping(metadata or {})
    return {
        "event_type": event_type,
        "operation": operation,
        "status": status,
        "duration_ms": _elapsed_ms(start) if duration_ms is None else max(0.0, round(float(duration_ms), 3)),
        "metadata": safe_metadata,
    }


def _mcp_metrics(events: Any) -> dict[str, Any]:
    return {"events": [dict(event) for event in events]}


def _tool_call_summary(
    tool_call_id: str,
    name: str,
    arguments: Mapping[str, Any],
    *,
    status: str,
) -> Mapping[str, object]:
    return {
        "tool_call_id": _sanitize(tool_call_id),
        "name": _sanitize(name),
        "arguments_summary": _arguments_summary(arguments),
        "status": status,
    }


def _arguments_summary(arguments: Mapping[str, Any]) -> dict[str, object]:
    summary: dict[str, object] = {}
    if arguments.get("query") is not None:
        summary["query"] = _sanitize(str(arguments.get("query")))
    if arguments.get("top_k") is not None:
        summary["top_k"] = arguments.get("top_k")
    if arguments.get("retrieval_mode") is not None:
        summary["retrieval_mode"] = _sanitize(str(arguments.get("retrieval_mode")))
    filters = arguments.get("filters")
    if isinstance(filters, Mapping):
        summary["filters_present"] = bool(filters)
        summary["filter_keys"] = sorted(_sanitize(str(key)) for key in filters.keys() if str(key) != "root")
    return summary


def _tool_results_summary(tool_name: str, tool_result: Mapping[str, Any]) -> Mapping[str, object]:
    rows = _result_rows(tool_result)
    return {
        "tool_name": _sanitize(tool_name),
        "result_count": len(rows),
        "fallback_used": bool(tool_result.get("fallback_used", False)),
        "fallback_reason": _sanitize(str(tool_result.get("fallback_reason")))
        if tool_result.get("fallback_reason") is not None
        else None,
        "source_doc_names": sorted(
            {str(row.get("source_doc_name", "")) for row in rows if row.get("source_doc_name")}
        ),
        "chunk_ids": [str(row.get("chunk_id", "")) for row in rows if row.get("chunk_id")],
    }


def _empty_tool_results_summary() -> Mapping[str, object]:
    return {
        "tool_name": SEARCH_KNOWLEDGE_BASE_TOOL,
        "result_count": 0,
        "fallback_used": False,
        "fallback_reason": None,
        "source_doc_names": [],
        "chunk_ids": [],
    }


def _raw_result_count(tool_result: Mapping[str, Any]) -> int:
    rows = tool_result.get("results", ())
    return len(rows) if isinstance(rows, list | tuple) else 0


def _sanitize_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _sanitize_jsonish(nested) for key, nested in value.items() if str(key) != "root"}


def _sanitize_jsonish(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _sanitize_mapping(value)
    if isinstance(value, list | tuple):
        return [_sanitize_jsonish(item) for item in value]
    if isinstance(value, str):
        return _sanitize(value)
    return value


def _has_no_event(events: list[dict[str, Any]], event_type: str) -> bool:
    return not any(event.get("event_type") == event_type for event in events)

class AskMCPToolLoopError(RuntimeError):
    """
    当 MCP 工具循环无法编排时抛出。

    """