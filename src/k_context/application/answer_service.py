"""Grounded answer synthesis over retrieved chunks."""

from __future__ import annotations

from dataclasses import dataclass

from k_context.application.retrieval_service import SearchResult


EVIDENCE_INSUFFICIENT = "证据不足"
EVIDENCE_BASIC = "基本充分"
EVIDENCE_SUFFICIENT = "充分"
EVIDENCE_LEVELS = (EVIDENCE_INSUFFICIENT, EVIDENCE_BASIC, EVIDENCE_SUFFICIENT)


@dataclass(frozen=True)
class AnswerSource:
    """Traceable source used by grounded answer output."""

    chunk_id: str
    source_doc_id: str
    source_doc_name: str
    score: float
    block_ids: tuple[str, ...]


@dataclass(frozen=True)
class GroundedAnswer:
    """Conservative answer generated only from retrieved chunks."""

    answer: str
    evidence_level: str
    sources: tuple[AnswerSource, ...]


class GroundedAnswerService:
    """Synthesizes a minimal answer without calling an external model."""

    def synthesize(self, question: str, search_results: tuple[SearchResult, ...]) -> GroundedAnswer:
        if not search_results:
            return GroundedAnswer(
                answer="当前知识库依据不足，无法回答该问题。",
                evidence_level=EVIDENCE_INSUFFICIENT,
                sources=(),
            )

        sources = tuple(
            AnswerSource(
                chunk_id=result.chunk.chunk_id,
                source_doc_id=result.chunk.source_doc_id,
                source_doc_name=result.chunk.source_doc_name,
                score=result.score,
                block_ids=result.chunk.block_ids,
            )
            for result in search_results
        )
        answer_parts = [
            "根据当前知识库中检索到的片段，保守回答如下：",
            f"问题：{question}",
        ]
        for index, result in enumerate(search_results, start=1):
            answer_parts.append(f"依据 {index}：{result.snippet()}")

        evidence_level = EVIDENCE_SUFFICIENT if len(search_results) > 1 else EVIDENCE_BASIC
        return GroundedAnswer(
            answer="\n".join(answer_parts),
            evidence_level=evidence_level,
            sources=sources,
        )
