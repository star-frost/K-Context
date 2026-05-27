"""应用层索引构建编排。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, TypeVar

from k_context.application.chunking_service import DSC_CHUNKING_STRATEGY, ChunkingService
from k_context.application.cleaning_service import CleaningService
from k_context.application.config_service import ConfigService
from k_context.application.embedding_provider import EmbeddingInput, EmbeddingProvider
from k_context.application.metrics_collector import MetricsCollector
from k_context.application.vector_store import VectorStore, VectorStoreUpsertResult
from k_context.domain.models import (
    Chunk,
    DocumentBlock,
    EmbeddingRecord,
    KContextConfig,
    VectorRecord,
)
from k_context.infrastructure.storage.local_store import LocalKnowledgeBaseStore


DEFAULT_INDEX_VERSION = "default"
T = TypeVar("T")


class IndexServiceError(RuntimeError):
    """当索引构建流水线失败且已记录指标事件时抛出。"""


@dataclass(frozen=True)
class IndexBuildResult:
    """IndexService 完成一次索引构建后返回的摘要。"""

    cleaned_blocks: tuple[DocumentBlock, ...]
    chunks: tuple[Chunk, ...]
    embedding_records: tuple[EmbeddingRecord, ...]
    vector_records: tuple[VectorRecord, ...]
    vector_upsert_result: VectorStoreUpsertResult
    cleaned_blocks_path: Path
    chunks_path: Path
    metrics_path: Path
    index_version: str
    rebuild: bool


class IndexService:
    """从持久化块开始，经向量 upsert 构建本地 RAG 索引。"""

    def __init__(
        self,
        *,
        store: LocalKnowledgeBaseStore | None = None,
        config_service: ConfigService | None = None,
        cleaning_service: CleaningService | None = None,
        chunking_service: ChunkingService | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        embedding_provider_factory: (
            Callable[[Path, KContextConfig], EmbeddingProvider] | None
        ) = None,
        vector_store: VectorStore | None = None,
        vector_store_factory: (
            Callable[[Path, KContextConfig, int | None, str], VectorStore] | None
        ) = None,
        metrics_collector_factory: Callable[[Path], MetricsCollector] | None = None,
        index_version: str = DEFAULT_INDEX_VERSION,
    ) -> None:
        self._store = store or LocalKnowledgeBaseStore()
        self._config_service = config_service or ConfigService()
        self._cleaning = cleaning_service or CleaningService()
        self._chunking = chunking_service or ChunkingService()
        self._embedding_provider = embedding_provider
        self._embedding_provider_factory = (
            embedding_provider_factory or self._default_embedding_provider_factory
        )
        self._vector_store = vector_store
        self._vector_store_factory = vector_store_factory or self._default_vector_store_factory
        self._metrics_collector_factory = metrics_collector_factory or MetricsCollector.from_root
        self._index_version = index_version

    def build(
        self,
        root: Path,
        *,
        runtime_overrides: Mapping[str, object | None] | None = None,
        rebuild: bool = False,
    ) -> IndexBuildResult:
        """执行 clean -> chunk -> embed -> vector upsert，并记录各步骤指标。"""

        kb_paths = self._store.require_initialized(root)
        config = self._config_service.load(root, runtime_overrides=runtime_overrides)
        metrics = self._metrics_collector_factory(root)
        total_started_at = _utc_now()
        total_start = perf_counter()
        base_metadata = self._base_metadata(root=root, config=config)

        try:
            block_records = self._store.read_block_records(kb_paths)
            blocks = tuple(DocumentBlock.from_dict(record) for record in block_records)

            cleaned_blocks = self._run_timed(
                metrics=metrics,
                event_type="cleaning",
                operation="cleaning_time",
                metadata={**base_metadata, "record_count": len(blocks)},
                action=lambda: self._cleaning.clean(
                    blocks,
                    cleaning_profile=config.cleaning_profile,
                ),
            )
            self._store.replace_cleaned_block_records(
                kb_paths,
                (block.to_dict() for block in cleaned_blocks),
            )

            embedding_provider: EmbeddingProvider | None = None
            if config.chunking_strategy == DSC_CHUNKING_STRATEGY:
                embedding_provider = self._get_embedding_provider(root, config)

            chunks = self._run_timed(
                metrics=metrics,
                event_type="chunking",
                operation="chunking_time",
                metadata={**base_metadata, "record_count": len(cleaned_blocks)},
                action=lambda: self._chunking.generate(
                    cleaned_blocks,
                    chunking_strategy=config.chunking_strategy,
                    embedding_provider=embedding_provider,
                ),
            )
            self._store.replace_records(kb_paths.chunks_path, (chunk.to_dict() for chunk in chunks))

            if embedding_provider is None:
                embedding_provider = self._get_embedding_provider(root, config)
            embedding_records = self._run_timed(
                metrics=metrics,
                event_type="embedding",
                operation="embedding_time",
                metadata={
                    **base_metadata,
                    "record_count": len(chunks),
                    "embedding_model": embedding_provider.embedding_model,
                    "embedding_dim": embedding_provider.embedding_dim,
                },
                action=lambda: embedding_provider.embed_documents(
                    tuple(
                        EmbeddingInput(source_id=chunk.chunk_id, text=chunk.text)
                        for chunk in chunks
                    )
                ),
            )
            vector_records = self._to_vector_records(
                chunks=chunks,
                embedding_records=embedding_records,
                config=config,
            )
            vector_store = self._get_vector_store(
                root=root,
                config=config,
                embedding_dim=embedding_provider.embedding_dim,
            )
            vector_upsert_result = self._run_timed(
                metrics=metrics,
                event_type="vector_upsert",
                operation="vector_upsert_time",
                metadata={
                    **base_metadata,
                    "record_count": len(vector_records),
                    "embedding_model": embedding_provider.embedding_model,
                    "embedding_dim": embedding_provider.embedding_dim,
                },
                action=lambda: (
                    vector_store.rebuild_collection(vector_records)
                    if rebuild
                    else vector_store.upsert(vector_records)
                ),
            )

            metrics.record_success(
                event_type="index",
                operation="index_total_time",
                started_at=total_started_at,
                ended_at=_utc_now(),
                duration_ms=_elapsed_ms(total_start),
                metadata={
                    **base_metadata,
                    "record_count": len(vector_records),
                    "rebuild": rebuild,
                },
            )
            return IndexBuildResult(
                cleaned_blocks=cleaned_blocks,
                chunks=chunks,
                embedding_records=embedding_records,
                vector_records=vector_records,
                vector_upsert_result=vector_upsert_result,
                cleaned_blocks_path=kb_paths.cleaned_blocks_path,
                chunks_path=kb_paths.chunks_path,
                metrics_path=kb_paths.metrics_path,
                index_version=self._index_version,
                rebuild=rebuild,
            )
        except Exception as exc:
            metrics.record_failure(
                event_type="index",
                operation="index_total_time",
                error_message=str(exc),
                started_at=total_started_at,
                ended_at=_utc_now(),
                duration_ms=_elapsed_ms(total_start),
                metadata={**base_metadata, "rebuild": rebuild},
            )
            if isinstance(exc, IndexServiceError):
                raise
            raise IndexServiceError(f"Index build failed: {exc}") from exc

    def _run_timed(
        self,
        *,
        metrics: MetricsCollector,
        event_type: str,
        operation: str,
        metadata: Mapping[str, Any],
        action: Callable[[], T],
    ) -> T:
        started_at = _utc_now()
        start = perf_counter()
        try:
            result = action()
        except Exception as exc:
            metrics.record_failure(
                event_type=event_type,
                operation=operation,
                error_message=str(exc),
                started_at=started_at,
                ended_at=_utc_now(),
                duration_ms=_elapsed_ms(start),
                metadata=metadata,
            )
            raise IndexServiceError(f"{operation} failed: {exc}") from exc
        metrics.record_success(
            event_type=event_type,
            operation=operation,
            started_at=started_at,
            ended_at=_utc_now(),
            duration_ms=_elapsed_ms(start),
            metadata=metadata,
        )
        return result

    def _to_vector_records(
        self,
        *,
        chunks: tuple[Chunk, ...],
        embedding_records: tuple[EmbeddingRecord, ...],
        config: KContextConfig,
    ) -> tuple[VectorRecord, ...]:
        records_by_chunk_id = {record.chunk_id: record for record in embedding_records}
        vector_records = []
        for chunk in chunks:
            try:
                embedding_record = records_by_chunk_id[chunk.chunk_id]
            except KeyError as exc:
                raise IndexServiceError(
                    f"Missing embedding record for chunk_id {chunk.chunk_id}."
                ) from exc
            vector_records.append(
                VectorRecord.create(
                    chunk=chunk,
                    embedding_record=embedding_record,
                    chunking_strategy=config.chunking_strategy,
                    cleaning_profile=config.cleaning_profile,
                    index_version=self._index_version,
                )
            )
        return tuple(vector_records)

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
        return self._vector_store_factory(root, config, embedding_dim, self._index_version)

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
        index_version: str,
    ) -> VectorStore:
        from k_context.infrastructure.vector_stores.chroma_store import ChromaVectorStore

        return ChromaVectorStore.from_config(
            root,
            config_service=self._config_service,
            config=config,
            embedding_dim=embedding_dim,
            index_version=index_version,
        )

    def _base_metadata(self, *, root: Path, config: KContextConfig) -> dict[str, Any]:
        return {
            "root": str(root.expanduser().resolve()),
            "chunking_strategy": config.chunking_strategy,
            "cleaning_profile": config.cleaning_profile,
            "embedding_model": config.embedding_model,
            "vector_store_type": config.vector_store_type,
            "retrieval_mode": config.retrieval_mode,
            "top_k": config.top_k,
            "index_version": self._index_version,
        }


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _elapsed_ms(start: float) -> float:
    return max(0.0, round((perf_counter() - start) * 1000, 3))
