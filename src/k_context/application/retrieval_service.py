"""基于持久化切块和向量存储索引的本地检索。"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

from k_context.application.config_service import ConfigService
from k_context.application.embedding_provider import EmbeddingProvider
from k_context.application.metrics_collector import MetricsCollector
from k_context.application.vector_store import VectorStore
from k_context.domain.models import Chunk, KContextConfig, VectorSearchResult
from k_context.infrastructure.storage.local_store import LocalKnowledgeBaseStore


DEFAULT_TOP_K = 5
SNIPPET_LENGTH = 160
TOKEN_PATTERN = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)
KEYWORD_RETRIEVAL_MODE = "keyword"
VECTOR_RETRIEVAL_MODE = "vector"
SUPPORTED_RETRIEVAL_MODES = {KEYWORD_RETRIEVAL_MODE, VECTOR_RETRIEVAL_MODE}
DEFAULT_INDEX_VERSION = "default"


class RetrievalServiceError(RuntimeError):
    """当检索无法完成且无兜底路径可用时抛出。"""


@dataclass(frozen=True)
class SearchResult:
    """绑定到一个 Chunk 的排序检索结果。"""

    chunk: Chunk
    score: float
    retrieval_mode: str = KEYWORD_RETRIEVAL_MODE
    metadata: dict[str, Any] = field(default_factory=dict)

    def snippet(self) -> str:
        text = self.chunk.text.strip()
        return text if len(text) <= SNIPPET_LENGTH else text[:SNIPPET_LENGTH].rstrip() + "..."

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id

    @property
    def source_doc_id(self) -> str:
        return self.chunk.source_doc_id

    @property
    def source_doc_name(self) -> str:
        return self.chunk.source_doc_name

    @property
    def block_ids(self) -> tuple[str, ...]:
        return self.chunk.block_ids

    @property
    def text(self) -> str:
        return self.chunk.text

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "source_doc_id": self.source_doc_id,
            "source_doc_name": self.source_doc_name,
            "score": self.score,
            "retrieval_mode": self.retrieval_mode,
            "block_ids": list(self.block_ids),
            "text": self.text,
            "metadata": dict(self.metadata),
        }


RetrievalHit = SearchResult | VectorSearchResult


@dataclass(frozen=True)
class RetrievalResults:
    """包含实际模式和兜底状态的应用层检索结果。"""

    results: tuple[RetrievalHit, ...]
    chunks_available: int
    requested_mode: str
    retrieval_mode: str
    top_k: int
    fallback_used: bool = False
    fallback_reason: str | None = None


class RetrievalService:
    """使用关键词或向量检索获取 top-k 切块。"""

    def __init__(
        self,
        *,
        store: LocalKnowledgeBaseStore | None = None,
        config_service: ConfigService | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        embedding_provider_factory: (
            Callable[[Path, KContextConfig], EmbeddingProvider] | None
        ) = None,
        vector_store: VectorStore | None = None,
        vector_store_factory: (
            Callable[[Path, KContextConfig, int | None], VectorStore] | None
        ) = None,
        metrics_collector_factory: Callable[[Path], MetricsCollector] | None = None,
        index_version: str = DEFAULT_INDEX_VERSION,
    ) -> None:
        self._store = store or LocalKnowledgeBaseStore()
        self._config_service = config_service or ConfigService()
        self._embedding_provider = embedding_provider
        self._embedding_provider_factory = (
            embedding_provider_factory or self._default_embedding_provider_factory
        )
        self._vector_store = vector_store
        self._vector_store_factory = vector_store_factory or self._default_vector_store_factory
        self._metrics_collector_factory = metrics_collector_factory or MetricsCollector.from_root
        self._index_version = index_version

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
        """从配置的关键词或向量路径检索 top-k 结果。"""

        overrides = dict(runtime_overrides or {})
        if mode is not None:
            overrides["retrieval_mode"] = mode
        if top_k is not None:
            overrides["top_k"] = top_k
        config = self._config_service.load(root, runtime_overrides=overrides)
        retrieval_mode = _normalize_retrieval_mode(config.retrieval_mode)
        metrics = self._metrics_collector_factory(root)

        if retrieval_mode == KEYWORD_RETRIEVAL_MODE:
            results, chunks_available = self._keyword_search_from_root(
                root=root,
                query=query,
                top_k=config.top_k,
                metrics=metrics,
                config=config,
                requested_mode=retrieval_mode,
                fallback_used=False,
                fallback_reason=None,
            )
            return RetrievalResults(
                results=results,
                chunks_available=chunks_available,
                requested_mode=retrieval_mode,
                retrieval_mode=KEYWORD_RETRIEVAL_MODE,
                top_k=config.top_k,
            )

        try:
            vector_results = self._vector_search(
                root=root,
                query=query,
                top_k=config.top_k,
                metrics=metrics,
                config=config,
            )
            return RetrievalResults(
                results=vector_results,
                chunks_available=len(vector_results),
                requested_mode=VECTOR_RETRIEVAL_MODE,
                retrieval_mode=VECTOR_RETRIEVAL_MODE,
                top_k=config.top_k,
            )
        except Exception as exc:
            if not allow_keyword_fallback:
                raise RetrievalServiceError(f"Vector retrieval failed: {exc}") from exc
            fallback_reason = str(exc)
            results, chunks_available = self._keyword_search_from_root(
                root=root,
                query=query,
                top_k=config.top_k,
                metrics=metrics,
                config=config,
                requested_mode=VECTOR_RETRIEVAL_MODE,
                fallback_used=True,
                fallback_reason=fallback_reason,
            )
            return RetrievalResults(
                results=results,
                chunks_available=chunks_available,
                requested_mode=VECTOR_RETRIEVAL_MODE,
                retrieval_mode=KEYWORD_RETRIEVAL_MODE,
                top_k=config.top_k,
                fallback_used=True,
                fallback_reason=fallback_reason,
            )

    def _keyword_search_from_root(
        self,
        *,
        root: Path,
        query: str,
        top_k: int,
        metrics: MetricsCollector,
        config: KContextConfig,
        requested_mode: str,
        fallback_used: bool,
        fallback_reason: str | None,
    ) -> tuple[tuple[SearchResult, ...], int]:
        kb_paths = self._store.require_initialized(root)
        chunk_records = self._store.read_records(kb_paths.chunks_path)
        chunks = tuple(Chunk.from_dict(record) for record in chunk_records)
        metadata = self._metrics_metadata(
            root=root,
            config=config,
            retrieval_mode=KEYWORD_RETRIEVAL_MODE,
            top_k=top_k,
            fallback_used=fallback_used,
            requested_mode=requested_mode,
            fallback_reason=fallback_reason,
            record_count=len(chunks),
        )
        started_at = _utc_now()
        start = perf_counter()
        try:
            results = self.search(chunks=chunks, query=query, top_k=top_k)
            results = tuple(
                SearchResult(
                    chunk=result.chunk,
                    score=result.score,
                    retrieval_mode=KEYWORD_RETRIEVAL_MODE,
                    metadata={
                        **result.metadata,
                        "retrieval_mode": KEYWORD_RETRIEVAL_MODE,
                        "requested_mode": requested_mode,
                        "fallback_used": fallback_used,
                        **({"fallback_reason": fallback_reason} if fallback_reason else {}),
                    },
                )
                for result in results
            )
        except Exception as exc:
            metrics.record_failure(
                event_type="retrieval",
                operation="retrieval_time",
                error_message=str(exc),
                started_at=started_at,
                ended_at=_utc_now(),
                duration_ms=_elapsed_ms(start),
                metadata=metadata,
            )
            raise
        metrics.record_success(
            event_type="retrieval",
            operation="retrieval_time",
            started_at=started_at,
            ended_at=_utc_now(),
            duration_ms=_elapsed_ms(start),
            metadata=metadata,
        )
        return results, len(chunks)

    def _vector_search(
        self,
        *,
        root: Path,
        query: str,
        top_k: int,
        metrics: MetricsCollector,
        config: KContextConfig,
    ) -> tuple[VectorSearchResult, ...]:
        embedding_provider = self._get_embedding_provider(root, config)
        embedding_metadata = self._metrics_metadata(
            root=root,
            config=config,
            retrieval_mode=VECTOR_RETRIEVAL_MODE,
            top_k=top_k,
            fallback_used=False,
            requested_mode=VECTOR_RETRIEVAL_MODE,
            record_count=1,
            embedding_model=embedding_provider.embedding_model,
            embedding_dim=embedding_provider.embedding_dim,
        )
        embedding_started_at = _utc_now()
        embedding_start = perf_counter()
        try:
            query_embedding = embedding_provider.embed_query(query)
        except Exception as exc:
            metrics.record_failure(
                event_type="retrieval",
                operation="query_embedding_time",
                error_message=str(exc),
                started_at=embedding_started_at,
                ended_at=_utc_now(),
                duration_ms=_elapsed_ms(embedding_start),
                metadata=embedding_metadata,
            )
            raise
        metrics.record_success(
            event_type="retrieval",
            operation="query_embedding_time",
            started_at=embedding_started_at,
            ended_at=_utc_now(),
            duration_ms=_elapsed_ms(embedding_start),
            metadata=embedding_metadata,
        )

        vector_store = self._get_vector_store(
            root=root,
            config=config,
            embedding_dim=query_embedding.embedding_dim,
        )
        retrieval_metadata = self._metrics_metadata(
            root=root,
            config=config,
            retrieval_mode=VECTOR_RETRIEVAL_MODE,
            top_k=top_k,
            fallback_used=False,
            requested_mode=VECTOR_RETRIEVAL_MODE,
            embedding_model=query_embedding.embedding_model,
            embedding_dim=query_embedding.embedding_dim,
        )
        retrieval_started_at = _utc_now()
        retrieval_start = perf_counter()
        try:
            results = vector_store.query(query_embedding, top_k=top_k)
        except Exception as exc:
            metrics.record_failure(
                event_type="retrieval",
                operation="retrieval_time",
                error_message=str(exc),
                started_at=retrieval_started_at,
                ended_at=_utc_now(),
                duration_ms=_elapsed_ms(retrieval_start),
                metadata=retrieval_metadata,
            )
            raise
        metrics.record_success(
            event_type="retrieval",
            operation="retrieval_time",
            started_at=retrieval_started_at,
            ended_at=_utc_now(),
            duration_ms=_elapsed_ms(retrieval_start),
            metadata={**retrieval_metadata, "record_count": len(results)},
        )
        return results

    def _get_embedding_provider(
        self,
        root: Path,
        config: KContextConfig,
    ) -> EmbeddingProvider:
        if self._embedding_provider is not None:
            return self._embedding_provider
        return self._embedding_provider_factory(root, config)

    def _get_vector_store(
        self,
        *,
        root: Path,
        config: KContextConfig,
        embedding_dim: int | None,
    ) -> VectorStore:
        if self._vector_store is not None:
            return self._vector_store
        return self._vector_store_factory(root, config, embedding_dim)

    def _default_embedding_provider_factory(
        self,
        root: Path,
        config: KContextConfig,
    ) -> EmbeddingProvider:
        del config
        from k_context.infrastructure.embedding.sentence_transformer_provider import (
            SentenceTransformerEmbeddingProvider,
        )

        return SentenceTransformerEmbeddingProvider.from_config(
            root,
            config_service=self._config_service,
        )

    def _default_vector_store_factory(
        self,
        root: Path,
        config: KContextConfig,
        embedding_dim: int | None,
    ) -> VectorStore:
        if config.vector_store_type != "chroma":
            raise RetrievalServiceError(
                f"Unsupported vector_store_type for vector retrieval: {config.vector_store_type}."
            )
        from k_context.infrastructure.vector_stores.chroma_store import ChromaVectorStore

        return ChromaVectorStore.from_config(
            root,
            config_service=self._config_service,
            config=config,
            embedding_dim=embedding_dim,
            index_version=self._index_version,
        )

    def _metrics_metadata(
        self,
        *,
        root: Path,
        config: KContextConfig,
        retrieval_mode: str,
        top_k: int,
        fallback_used: bool,
        requested_mode: str,
        fallback_reason: str | None = None,
        record_count: int | None = None,
        embedding_model: str | None = None,
        embedding_dim: int | None = None,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "root": str(root.expanduser().resolve()),
            "chunking_strategy": config.chunking_strategy,
            "cleaning_profile": config.cleaning_profile,
            "embedding_model": embedding_model or config.embedding_model,
            "vector_store_type": config.vector_store_type,
            "retrieval_mode": retrieval_mode,
            "requested_mode": requested_mode,
            "top_k": top_k,
            "fallback_used": fallback_used,
            "index_version": self._index_version,
        }
        if fallback_reason:
            metadata["fallback_reason"] = fallback_reason
        if record_count is not None:
            metadata["record_count"] = record_count
        if embedding_dim is not None:
            metadata["embedding_dim"] = embedding_dim
        return metadata

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


def _normalize_retrieval_mode(value: str) -> str:
    mode = str(value).strip().casefold()
    if mode not in SUPPORTED_RETRIEVAL_MODES:
        raise ValueError(
            "Unsupported retrieval_mode: "
            f"{value}. Supported values: {', '.join(sorted(SUPPORTED_RETRIEVAL_MODES))}."
        )
    return mode


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _elapsed_ms(start: float) -> float:
    return max(0.0, round((perf_counter() - start) * 1000, 3))
