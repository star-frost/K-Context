"""基于检索切块的有依据回答合成。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping
from typing import Protocol

from k_context.application.llm_client import TOKEN_USAGE_UNAVAILABLE, TokenUsage


EVIDENCE_INSUFFICIENT = "证据不足"
EVIDENCE_BASIC = "基本充分"
EVIDENCE_SUFFICIENT = "充分"
EVIDENCE_LEVELS = (EVIDENCE_INSUFFICIENT, EVIDENCE_BASIC, EVIDENCE_SUFFICIENT)


@dataclass(frozen=True)
class AnswerSource:
    """有依据回答输出使用的可追溯来源。"""

    chunk_id: str
    source_doc_id: str
    source_doc_name: str
    score: float
    block_ids: tuple[str, ...]
    retrieval_mode: str = "keyword"
    page_start: int | None = None
    page_end: int | None = None
    quote: str | None = None
    fallback_used: bool = False
    fallback_reason: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class GroundedAnswer:
    """仅基于检索切块生成的保守回答。"""

    answer: str
    evidence_level: str
    sources: tuple[AnswerSource, ...]
    retrieval_mode: str = "keyword"
    top_k: int | None = None
    fallback_used: bool = False
    fallback_reason: str | None = None
    token_usage: TokenUsage = field(default_factory=TokenUsage.unavailable)
    token_usage_source: str = TOKEN_USAGE_UNAVAILABLE
    latency_ms: float = 0.0
    rag_method: str = "standard"
    deep_rag_steps: tuple[Mapping[str, object], ...] = ()
    tool_calls_used: bool = False
    mcp_server_transport: str | None = None
    tool_name: str | None = None
    tool_loop_count: int = 0
    tool_calls: tuple[Mapping[str, object], ...] = ()
    tool_results_summary: Mapping[str, object] = field(default_factory=dict)
    mcp_metrics: Mapping[str, object] = field(default_factory=dict)


class GroundedSearchResult(Protocol):
    """有依据回答合成所需的搜索结果字段。"""

    chunk_id: str
    source_doc_id: str
    source_doc_name: str
    score: float
    retrieval_mode: str
    block_ids: tuple[str, ...]
    text: str
    metadata: dict[str, object]

    def snippet(self) -> str:
        """返回适合展示的文本片段。"""


class GroundedAnswerService:
    """不调用外部模型，合成最小回答。"""

    def synthesize(
        self,
        question: str,
        search_results: tuple[GroundedSearchResult, ...],
        *,
        retrieval_mode: str = "keyword",
        top_k: int | None = None,
        fallback_used: bool = False,
        fallback_reason: str | None = None,
    ) -> GroundedAnswer:
        if not search_results:
            return GroundedAnswer(
                answer="当前知识库依据不足，无法回答该问题。",
                evidence_level=EVIDENCE_INSUFFICIENT,
                sources=(),
                retrieval_mode=retrieval_mode,
                top_k=top_k,
                fallback_used=fallback_used,
                fallback_reason=fallback_reason,
            )

        sources = tuple(
            AnswerSource(
                chunk_id=result.chunk_id,
                source_doc_id=result.source_doc_id,
                source_doc_name=result.source_doc_name,
                score=result.score,
                block_ids=result.block_ids,
                retrieval_mode=result.retrieval_mode,
                page_start=_result_page_value(result, "page_start"),
                page_end=_result_page_value(result, "page_end"),
                quote=_snippet(result),
                fallback_used=bool(result.metadata.get("fallback_used", fallback_used)),
                fallback_reason=(
                    str(result.metadata["fallback_reason"])
                    if result.metadata.get("fallback_reason") is not None
                    else fallback_reason
                ),
            )
            for result in search_results
        )
        answer_parts = [
            "根据当前知识库中检索到的片段，保守回答如下：",
            f"问题：{question}",
        ]
        for index, result in enumerate(search_results, start=1):
            answer_parts.append(f"依据 {index}：{_snippet(result)}")

        evidence_level = EVIDENCE_SUFFICIENT if len(search_results) > 1 else EVIDENCE_BASIC
        return GroundedAnswer(
            answer="\n".join(answer_parts),
            evidence_level=evidence_level,
            sources=sources,
            retrieval_mode=retrieval_mode,
            top_k=top_k,
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
        )


def _snippet(result: GroundedSearchResult) -> str:
    snippet = getattr(result, "snippet", None)
    if callable(snippet):
        return str(snippet())
    text = result.text.strip()
    return text if len(text) <= 160 else text[:160].rstrip() + "..."


def _result_page_value(result: GroundedSearchResult, field_name: str) -> int | None:
    chunk = getattr(result, "chunk", None)
    values = (
        getattr(chunk, field_name, None) if chunk is not None else None,
        result.metadata.get(field_name),
    )
    for value in values:
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    return None
