"""应用层 MCP 工具实现。

本模块中的类只作为可调用工具包装器。它们不暴露 stdio 传输，
不实现 tools/list 或 tools/call，也不调用 LLM。
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

from k_context.application.mcp_contracts import (
    SearchKnowledgeBaseInput,
    SearchKnowledgeBaseMetrics,
    SearchKnowledgeBaseOutput,
    SearchKnowledgeBaseResultItem,
)
from k_context.application.retrieval_service import RetrievalService


class SearchKnowledgeBaseToolError(RuntimeError):
    """当 search_knowledge_base 无法完成检索调用时抛出。"""


class SearchKnowledgeBaseTool:
    """搜索构造时绑定的知识库。

    绑定的 root 有意作为构造状态，而不是工具输入。未来的
    MCP server 代码应在 server 进程绑定到知识库 root 后
    再实例化此工具。
    """

    def __init__(
        self,
        *,
        root: Path,
        retrieval_service: RetrievalService | None = None,
    ) -> None:
        self._root = root.expanduser().resolve()
        self._retrieval_service = retrieval_service or RetrievalService()

    @property
    def root(self) -> Path:
        """返回绑定到此工具实例的知识库 root。"""

        return self._root

    def call(
        self,
        tool_input: SearchKnowledgeBaseInput | Mapping[str, Any],
    ) -> SearchKnowledgeBaseOutput:
        """运行检索并将结果转换为文档规定的 MCP 载荷。"""

        parsed_input = (
            tool_input
            if isinstance(tool_input, SearchKnowledgeBaseInput)
            else SearchKnowledgeBaseInput.from_dict(tool_input)
        )
        started = perf_counter()
        try:
            retrieval_result = self._retrieval_service.retrieve(
                root=self._root,
                query=parsed_input.query,
                top_k=parsed_input.top_k,
                mode=parsed_input.retrieval_mode,
                runtime_overrides=_runtime_overrides(parsed_input),
            )
        except Exception as exc:
            raise SearchKnowledgeBaseToolError(
                f"search_knowledge_base retrieval failed: {exc.__class__.__name__}: {exc}"
            ) from exc

        elapsed_ms = max(0.0, round((perf_counter() - started) * 1000.0, 3))
        return SearchKnowledgeBaseOutput(
            results=tuple(_result_item(result) for result in retrieval_result.results),
            fallback_used=bool(retrieval_result.fallback_used),
            fallback_reason=retrieval_result.fallback_reason,
            metrics=SearchKnowledgeBaseMetrics(
                retrieval_time_ms=elapsed_ms,
                result_count=len(retrieval_result.results),
                extra={
                    "requested_mode": retrieval_result.requested_mode,
                    "retrieval_mode": retrieval_result.retrieval_mode,
                    "top_k": retrieval_result.top_k,
                    "chunks_available": retrieval_result.chunks_available,
                },
            ),
        )


def _runtime_overrides(tool_input: SearchKnowledgeBaseInput) -> dict[str, object]:
    overrides: dict[str, object] = {}
    if tool_input.top_k is not None:
        overrides["top_k"] = tool_input.top_k
    if tool_input.retrieval_mode is not None:
        overrides["retrieval_mode"] = tool_input.retrieval_mode
    if tool_input.filters:
        overrides["filters"] = dict(tool_input.filters)
    return overrides


def _result_item(result: Any) -> SearchKnowledgeBaseResultItem:
    metadata = dict(getattr(result, "metadata", {}) or {})
    return SearchKnowledgeBaseResultItem(
        chunk_id=str(getattr(result, "chunk_id")),
        source_doc_id=str(getattr(result, "source_doc_id")),
        source_doc_name=str(getattr(result, "source_doc_name")),
        score=float(getattr(result, "score")),
        retrieval_mode=str(getattr(result, "retrieval_mode")),
        block_ids=tuple(str(item) for item in getattr(result, "block_ids")),
        page_start=_page_value(result, "page_start"),
        page_end=_page_value(result, "page_end"),
        quote=_quote(result),
        metadata=metadata,
    )


def _page_value(result: Any, field_name: str) -> int | None:
    direct_value = getattr(result, field_name, None)
    metadata = getattr(result, "metadata", {}) or {}
    values = (direct_value, metadata.get(field_name))
    chunk = getattr(result, "chunk", None)
    if chunk is not None:
        values = (direct_value, getattr(chunk, field_name, None), metadata.get(field_name))
    for value in values:
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    return None


def _quote(result: Any) -> str | None:
    snippet = getattr(result, "snippet", None)
    if callable(snippet):
        quote = str(snippet()).strip()
        return quote or None
    text = str(getattr(result, "text", "")).strip()
    if not text:
        return None
    return text if len(text) <= 300 else text[:300].rstrip() + "..."
