"""Minimal local retrieval over persisted Chunk records."""

from __future__ import annotations

import re
from dataclasses import dataclass

from k_context.domain.models import Chunk


DEFAULT_TOP_K = 5
SNIPPET_LENGTH = 160
TOKEN_PATTERN = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)


@dataclass(frozen=True)
class SearchResult:
    """Ranked retrieval result tied to one Chunk."""

    chunk: Chunk
    score: float

    def snippet(self) -> str:
        text = self.chunk.text.strip()
        return text if len(text) <= SNIPPET_LENGTH else text[:SNIPPET_LENGTH].rstrip() + "..."


class RetrievalService:
    """Scores user queries against Chunk text using lightweight local matching."""

    def search(
        self,
        *,
        chunks: tuple[Chunk, ...],
        query: str,
        top_k: int = DEFAULT_TOP_K,
    ) -> tuple[SearchResult, ...]:
        query_terms = self._tokenize(query)
        if not chunks or not query_terms or top_k <= 0:
            return ()

        scored_results: list[SearchResult] = []
        for chunk in chunks:
            score = self._score(chunk, query_terms)
            if score > 0:
                scored_results.append(SearchResult(chunk=chunk, score=score))

        return tuple(
            sorted(scored_results, key=lambda item: (-item.score, item.chunk.chunk_id))[:top_k]
        )

    def _score(self, chunk: Chunk, query_terms: tuple[str, ...]) -> float:
        text_terms = self._tokenize(chunk.text)
        if not text_terms:
            return 0.0

        text_term_counts: dict[str, int] = {}
        for term in text_terms:
            text_term_counts[term] = text_term_counts.get(term, 0) + 1

        return float(sum(text_term_counts.get(term, 0) for term in query_terms))

    def _tokenize(self, text: str) -> tuple[str, ...]:
        return tuple(match.group(0).casefold() for match in TOKEN_PATTERN.finditer(text))
