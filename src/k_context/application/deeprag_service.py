"""DeepRAG 启发式逐步检索编排。"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Protocol

from k_context.application.answer_service import GroundedAnswer, GroundedAnswerService
from k_context.application.llm_client import LLMClient, LLMClientError, LLMClientUnavailableError
from k_context.application.prompt_builder import PromptBuilder, PromptBuilderError
from k_context.application.retrieval_service import DEFAULT_TOP_K, RetrievalHit, RetrievalResults


STANDARD_RAG_METHOD = "standard"
DEEPRAG_RAG_METHOD = "deeprag"
SUPPORTED_RAG_METHODS = {STANDARD_RAG_METHOD, DEEPRAG_RAG_METHOD}
DEEPRAG_MAX_STEPS = 3

_SUBQUERY_SPLIT_PATTERN = re.compile(r"[?？;；\n]+|\s+\b(?:and|then)\b\s+", re.IGNORECASE)


class DeepRAGRetrievalService(Protocol):
    def retrieve(
        self,
        *,
        root: Path,
        query: str,
        top_k: int | None = None,
        mode: str | None = None,
        runtime_overrides: Mapping[str, object | None] | None = None,
        allow_keyword_fallback: bool = True,
    ) -> RetrievalResults:
        """为一个 DeepRAG 子查询检索证据。"""


@dataclass(frozen=True)
class DeepRAGRunResult:
    answer: GroundedAnswer
    requested_mode: str
    retrieval_mode: str
    top_k: int


class DeepRAGService:
    """用于对比实验的轻量 DeepRAG 风格方法。

    DeepRAG 论文将 RAG 建模为逐步问题分解加原子级检索/参数化决策。
    本地实现会对每个生成的子查询进行检索、聚合证据，
    并仅在最终有依据回答阶段使用 LLM，从而保持回答有据可依。
    """

    def __init__(
        self,
        *,
        retrieval_service: DeepRAGRetrievalService,
        answer_service: GroundedAnswerService | None = None,
        prompt_builder: PromptBuilder | None = None,
        max_steps: int = DEEPRAG_MAX_STEPS,
    ) -> None:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive.")
        self._retrieval = retrieval_service
        self._answer_service = answer_service or GroundedAnswerService()
        self._prompt_builder = prompt_builder or PromptBuilder()
        self._max_steps = max_steps

    def run(
        self,
        *,
        root: Path,
        question: str,
        top_k: int | None = None,
        mode: str | None = None,
        runtime_overrides: Mapping[str, object | None] | None = None,
        use_llm: bool,
        llm_client: LLMClient | None = None,
    ) -> DeepRAGRunResult:
        query = question.strip()
        if not query:
            raise ValueError("question must not be empty.")
        effective_top_k = int(top_k or (runtime_overrides or {}).get("top_k") or DEFAULT_TOP_K)
        subqueries = _subqueries(query, max_steps=self._max_steps)
        steps: list[dict[str, object]] = []
        aggregated = _aggregate_results(
            self._retrieve_step(
                root=root,
                subquery=subquery,
                top_k=effective_top_k,
                mode=mode,
                runtime_overrides=runtime_overrides,
                step_index=index,
                steps=steps,
            )
            for index, subquery in enumerate(subqueries, start=1)
        )[:effective_top_k]
        retrieval_mode = _effective_retrieval_mode(aggregated, mode)
        requested_mode = str(mode or (runtime_overrides or {}).get("retrieval_mode") or retrieval_mode)

        grounded_answer = self._answer_service.synthesize(
            query,
            tuple(aggregated),
            retrieval_mode=retrieval_mode,
            top_k=effective_top_k,
            fallback_used=not use_llm,
            fallback_reason="no_llm_requested" if not use_llm else None,
        )
        answer = _with_deeprag(grounded_answer, steps)

        if use_llm and aggregated and llm_client is None:
            answer = _with_deeprag(
                self._answer_service.synthesize(
                    query,
                    tuple(aggregated),
                    retrieval_mode=retrieval_mode,
                    top_k=effective_top_k,
                    fallback_used=True,
                    fallback_reason="deeprag_llm_unavailable",
                ),
                steps,
            )
        elif use_llm and aggregated and llm_client is not None:
            llm_start = perf_counter()
            try:
                llm_request = self._prompt_builder.build(
                    question=query,
                    sources=tuple(aggregated),
                    system_instruction=(
                        "You are a grounded DeepRAG-style assistant. Answer only from the provided "
                        "sources gathered through step-by-step retrieval. If evidence is insufficient, say so."
                    ),
                )
                llm_response = llm_client.generate(llm_request)
                answer = GroundedAnswer(
                    answer=llm_response.answer,
                    evidence_level=grounded_answer.evidence_level,
                    sources=grounded_answer.sources,
                    retrieval_mode=retrieval_mode,
                    top_k=effective_top_k,
                    fallback_used=False,
                    fallback_reason=None,
                    token_usage=llm_response.token_usage,
                    token_usage_source=llm_response.token_usage_source,
                    latency_ms=llm_response.latency_ms,
                    rag_method=DEEPRAG_RAG_METHOD,
                    deep_rag_steps=tuple(steps),
                )
            except (LLMClientError, LLMClientUnavailableError, PromptBuilderError, ValueError) as exc:
                answer = _with_deeprag(
                    self._answer_service.synthesize(
                        query,
                        tuple(aggregated),
                        retrieval_mode=retrieval_mode,
                        top_k=effective_top_k,
                        fallback_used=True,
                        fallback_reason=f"deeprag_llm_failed:{_safe_reason(exc)}",
                    ),
                    steps,
                    latency_ms=_elapsed_ms(llm_start),
                )

        return DeepRAGRunResult(
            answer=answer,
            requested_mode=requested_mode,
            retrieval_mode=retrieval_mode,
            top_k=effective_top_k,
        )

    def _retrieve_step(
        self,
        *,
        root: Path,
        subquery: str,
        top_k: int,
        mode: str | None,
        runtime_overrides: Mapping[str, object | None] | None,
        step_index: int,
        steps: list[dict[str, object]],
    ) -> tuple[RetrievalHit, ...]:
        result = self._retrieval.retrieve(
            root=root,
            query=subquery,
            top_k=top_k,
            mode=mode,
            runtime_overrides=runtime_overrides,
        )
        steps.append(
            {
                "step": step_index,
                "subquery": subquery,
                "action": "retrieve",
                "retrieval_mode": result.retrieval_mode,
                "requested_mode": result.requested_mode,
                "top_k": result.top_k,
                "result_count": len(result.results),
                "retrieved_chunk_ids": [item.chunk_id for item in result.results],
                "fallback_used": result.fallback_used,
                "fallback_reason": result.fallback_reason,
            }
        )
        return result.results


def normalize_rag_method(value: str | None) -> str:
    method = (value or STANDARD_RAG_METHOD).strip().casefold()
    if method not in SUPPORTED_RAG_METHODS:
        supported = ", ".join(sorted(SUPPORTED_RAG_METHODS))
        raise ValueError(f"Unsupported rag_method: {value}. Supported methods: {supported}")
    return method


def _subqueries(question: str, *, max_steps: int) -> tuple[str, ...]:
    parts = tuple(part.strip(" .。?？") for part in _SUBQUERY_SPLIT_PATTERN.split(question) if part.strip())
    ordered: list[str] = []
    for part in parts:
        if part and part not in ordered:
            ordered.append(part)
    if question not in ordered:
        ordered.append(question)
    return tuple(ordered[:max_steps])


def _aggregate_results(result_groups: object) -> list[RetrievalHit]:
    by_chunk_id: dict[str, RetrievalHit] = {}
    for results in result_groups:
        for item in results:
            existing = by_chunk_id.get(item.chunk_id)
            if existing is None or item.score > existing.score:
                by_chunk_id[item.chunk_id] = item
    return sorted(by_chunk_id.values(), key=lambda item: (-item.score, item.chunk_id))


def _effective_retrieval_mode(results: list[RetrievalHit], requested_mode: str | None) -> str:
    modes = {item.retrieval_mode for item in results}
    if len(modes) == 1:
        return next(iter(modes))
    return requested_mode or "mixed"


def _with_deeprag(
    answer: GroundedAnswer,
    steps: list[dict[str, object]],
    *,
    latency_ms: float | None = None,
) -> GroundedAnswer:
    return GroundedAnswer(
        answer=answer.answer,
        evidence_level=answer.evidence_level,
        sources=answer.sources,
        retrieval_mode=answer.retrieval_mode,
        top_k=answer.top_k,
        fallback_used=answer.fallback_used,
        fallback_reason=answer.fallback_reason,
        token_usage=answer.token_usage,
        token_usage_source=answer.token_usage_source,
        latency_ms=answer.latency_ms if latency_ms is None else latency_ms,
        rag_method=DEEPRAG_RAG_METHOD,
        deep_rag_steps=tuple(steps),
    )


def _safe_reason(exc: Exception) -> str:
    return str(exc).strip() or exc.__class__.__name__


def _elapsed_ms(start: float) -> float:
    return max(0.0, round((perf_counter() - start) * 1000, 3))
