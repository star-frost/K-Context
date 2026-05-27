"""基于 Chroma 的本地持久化向量存储。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

from k_context.application.config_service import ConfigService
from k_context.application.vector_store import (
    VECTOR_RETRIEVAL_MODE,
    VectorStoreDimensionError,
    VectorStoreError,
    VectorStoreUpsertResult,
)
from k_context.domain.models import KContextConfig, QueryEmbedding, VectorRecord, VectorSearchResult


DEFAULT_INDEX_VERSION = "default"


@dataclass(frozen=True)
class ChromaPipelineConfig:
    """用于隔离一个 Chroma collection 的流水线标识。"""

    chunking_strategy: str
    cleaning_profile: str
    embedding_model: str
    index_version: str
    embedding_dim: int | None = None

    def collection_name(self) -> str:
        identity = "|".join(
            (
                self.chunking_strategy,
                self.cleaning_profile,
                self.embedding_model,
                self.index_version,
            )
        )
        return f"kcontext_{sha256(identity.encode('utf-8')).hexdigest()[:24]}"

    def metadata(self, *, embedding_dim: int, created_at: str) -> dict[str, Any]:
        return {
            "chunking_strategy": self.chunking_strategy,
            "cleaning_profile": self.cleaning_profile,
            "embedding_model": self.embedding_model,
            "embedding_dim": embedding_dim,
            "index_version": self.index_version,
            "created_at": created_at,
        }


class ChromaVectorStore:
    """由本地持久化 Chroma collection 支撑的 VectorStore 实现。"""

    def __init__(
        self,
        *,
        persist_dir: Path,
        chunking_strategy: str,
        cleaning_profile: str,
        embedding_model: str,
        index_version: str = DEFAULT_INDEX_VERSION,
        embedding_dim: int | None = None,
        collection_name: str | None = None,
        client_factory: Callable[[Path], Any] | None = None,
    ) -> None:
        if embedding_dim is not None and embedding_dim <= 0:
            raise ValueError("embedding_dim must be a positive integer.")
        self.persist_dir = persist_dir.expanduser().resolve()
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.pipeline = ChromaPipelineConfig(
            chunking_strategy=chunking_strategy,
            cleaning_profile=cleaning_profile,
            embedding_model=embedding_model,
            index_version=index_version,
            embedding_dim=embedding_dim,
        )
        self.collection_name = collection_name or self.pipeline.collection_name()
        self._embedding_dim = embedding_dim
        self._collection_created_at = datetime.now(timezone.utc).isoformat()
        self._client_factory = client_factory or self._default_client_factory
        self._client = self._create_client()
        self._collection: Any | None = None

    @classmethod
    def from_config(
        cls,
        root: Path,
        *,
        config_service: ConfigService | None = None,
        config: KContextConfig | None = None,
        chroma_persist_dir: Path | None = None,
        chunking_strategy: str | None = None,
        cleaning_profile: str | None = None,
        embedding_model: str | None = None,
        embedding_dim: int | None = None,
        index_version: str = DEFAULT_INDEX_VERSION,
        client_factory: Callable[[Path], Any] | None = None,
    ) -> "ChromaVectorStore":
        service = config_service or ConfigService()
        resolved_config = config or service.load(root)
        persist_dir = chroma_persist_dir or service.resolve_chroma_persist_dir(
            root,
            resolved_config,
        )
        return cls(
            persist_dir=persist_dir,
            chunking_strategy=chunking_strategy or resolved_config.chunking_strategy,
            cleaning_profile=cleaning_profile or resolved_config.cleaning_profile,
            embedding_model=embedding_model or resolved_config.embedding_model,
            embedding_dim=embedding_dim,
            index_version=index_version,
            client_factory=client_factory,
        )

    def upsert(self, records: tuple[VectorRecord, ...]) -> VectorStoreUpsertResult:
        if not records:
            return VectorStoreUpsertResult(
                collection_name=self.collection_name,
                upserted_count=0,
                replaced_chunk_ids=(),
            )
        self._validate_records(records)
        collection = self._get_or_create_collection(embedding_dim=records[0].embedding_dim)
        existing = collection.get(ids=[record.chunk_id for record in records], include=[])
        replaced_chunk_ids = tuple(str(chunk_id) for chunk_id in existing.get("ids", ()))
        try:
            collection.upsert(
                ids=[record.chunk_id for record in records],
                embeddings=[list(record.embedding) for record in records],
                metadatas=[self._metadata_for_record(record) for record in records],
                documents=[record.text for record in records],
            )
        except Exception as exc:  # pragma: no cover - exact Chroma exceptions vary by version
            raise VectorStoreError(f"Failed to upsert records into Chroma: {exc}") from exc
        return VectorStoreUpsertResult(
            collection_name=self.collection_name,
            upserted_count=len(records),
            replaced_chunk_ids=replaced_chunk_ids,
        )

    def query(
        self,
        query_embedding: QueryEmbedding,
        *,
        top_k: int,
    ) -> tuple[VectorSearchResult, ...]:
        if top_k <= 0:
            return ()
        self._validate_query(query_embedding)
        collection = self._get_collection_if_exists()
        if collection is None:
            return ()
        try:
            raw = collection.query(
                query_embeddings=[list(query_embedding.embedding)],
                n_results=top_k,
                include=["metadatas", "documents", "distances"],
            )
        except Exception as exc:  # pragma: no cover - exact Chroma exceptions vary by version
            raise VectorStoreError(f"Failed to query Chroma collection: {exc}") from exc
        return self._to_search_results(raw)

    def delete_collection(self) -> None:
        try:
            self._client.delete_collection(name=self.collection_name)
        except Exception as exc:
            if not self._is_not_found_error(exc):
                raise VectorStoreError(f"Failed to delete Chroma collection: {exc}") from exc
        self._collection = None
        self._embedding_dim = self.pipeline.embedding_dim
        self._collection_created_at = datetime.now(timezone.utc).isoformat()

    def rebuild_collection(self, records: tuple[VectorRecord, ...] = ()) -> VectorStoreUpsertResult:
        self.delete_collection()
        return self.upsert(records)

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            close()

    @property
    def collection_metadata(self) -> dict[str, Any] | None:
        collection = self._get_collection_if_exists()
        if collection is None:
            return None
        metadata = getattr(collection, "metadata", None)
        return dict(metadata or {})

    def _default_client_factory(self, persist_dir: Path) -> Any:
        try:
            import chromadb
        except ImportError as exc:
            raise VectorStoreError(
                "chromadb is not installed. Install the project dependency `chromadb` "
                "before using ChromaVectorStore."
            ) from exc
        return chromadb.PersistentClient(path=str(persist_dir))

    def _create_client(self) -> Any:
        try:
            return self._client_factory(self.persist_dir)
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError(f"Failed to initialize Chroma client: {exc}") from exc

    def _get_or_create_collection(self, *, embedding_dim: int) -> Any:
        self._set_or_check_dimension(embedding_dim)
        expected = self.pipeline.metadata(
            embedding_dim=embedding_dim,
            created_at=self._collection_created_at,
        )
        if self._collection is not None:
            self._assert_collection_metadata(self._collection, expected)
            return self._collection
        existing = self._get_collection_if_exists()
        if existing is not None:
            self._assert_collection_metadata(existing, expected)
            return existing
        try:
            collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata=expected,
                embedding_function=None,
            )
        except Exception as exc:
            raise VectorStoreError(f"Failed to open Chroma collection: {exc}") from exc
        self._assert_collection_metadata(collection, expected)
        self._collection = collection
        return collection

    def _get_collection_if_exists(self) -> Any | None:
        if self._collection is not None:
            return self._collection
        try:
            collection = self._client.get_collection(
                name=self.collection_name,
                embedding_function=None,
            )
        except Exception as exc:
            if self._is_not_found_error(exc):
                return None
            raise VectorStoreError(f"Failed to open Chroma collection: {exc}") from exc
        if collection.metadata and "embedding_dim" in collection.metadata:
            self._set_or_check_dimension(int(collection.metadata["embedding_dim"]))
        self._collection = collection
        return collection

    def _assert_collection_metadata(self, collection: Any, expected: dict[str, Any]) -> None:
        actual = dict(getattr(collection, "metadata", None) or {})
        for key, expected_value in expected.items():
            if key == "created_at":
                if key not in actual:
                    raise VectorStoreError(
                        "Chroma collection metadata missing required created_at."
                    )
                continue
            if actual.get(key) != expected_value:
                raise VectorStoreError(
                    f"Chroma collection metadata mismatch for {key}: "
                    f"expected {expected_value!r}, got {actual.get(key)!r}."
                )

    def _metadata_for_record(self, record: VectorRecord) -> dict[str, Any]:
        metadata = record.search_metadata()
        metadata["block_ids"] = list(record.block_ids)
        return metadata

    def _validate_records(self, records: tuple[VectorRecord, ...]) -> None:
        dimensions = {record.embedding_dim for record in records}
        if len(dimensions) != 1:
            raise VectorStoreDimensionError(
                "All VectorRecord values must have the same embedding_dim."
            )
        for record in records:
            if len(record.embedding) != record.embedding_dim:
                raise VectorStoreDimensionError(
                    "VectorRecord embedding length must match embedding_dim."
                )
            self._assert_record_pipeline(record)
        self._set_or_check_dimension(records[0].embedding_dim)

    def _assert_record_pipeline(self, record: VectorRecord) -> None:
        mismatches = []
        for field_name in (
            "chunking_strategy",
            "cleaning_profile",
            "embedding_model",
            "index_version",
        ):
            if getattr(record, field_name) != getattr(self.pipeline, field_name):
                mismatches.append(field_name)
        if mismatches:
            raise VectorStoreError(
                "VectorRecord pipeline metadata does not match Chroma collection: "
                + ", ".join(mismatches)
            )

    def _validate_query(self, query_embedding: QueryEmbedding) -> None:
        if len(query_embedding.embedding) != query_embedding.embedding_dim:
            raise VectorStoreDimensionError("Query embedding length must match embedding_dim.")
        self._set_or_check_dimension(query_embedding.embedding_dim)
        if query_embedding.embedding_model != self.pipeline.embedding_model:
            raise VectorStoreError(
                "Query embedding_model does not match Chroma collection: "
                f"expected {self.pipeline.embedding_model!r}, "
                f"got {query_embedding.embedding_model!r}."
            )

    def _set_or_check_dimension(self, embedding_dim: int) -> None:
        if embedding_dim <= 0:
            raise VectorStoreDimensionError("embedding_dim must be a positive integer.")
        if self._embedding_dim is None:
            self._embedding_dim = embedding_dim
            return
        if self._embedding_dim != embedding_dim:
            raise VectorStoreDimensionError(
                f"Vector dimension mismatch: expected {self._embedding_dim}, got {embedding_dim}."
            )

    def _to_search_results(self, raw: dict[str, Any]) -> tuple[VectorSearchResult, ...]:
        ids = self._first_query_values(raw, "ids")
        metadatas = self._first_query_values(raw, "metadatas")
        documents = self._first_query_values(raw, "documents")
        distances = self._first_query_values(raw, "distances")
        results: list[VectorSearchResult] = []
        for index, chunk_id in enumerate(ids):
            metadata = dict(metadatas[index] or {})
            block_ids = metadata.get("block_ids", ())
            if not isinstance(block_ids, (list, tuple)):
                block_ids = ()
            distance = float(distances[index]) if index < len(distances) else 0.0
            results.append(
                VectorSearchResult(
                    chunk_id=str(metadata.get("chunk_id") or chunk_id),
                    source_doc_id=str(metadata.get("source_doc_id") or ""),
                    source_doc_name=str(metadata.get("source_doc_name") or ""),
                    score=self._score_from_distance(distance),
                    retrieval_mode=VECTOR_RETRIEVAL_MODE,
                    block_ids=tuple(str(item) for item in block_ids),
                    text=str(documents[index] or "") if index < len(documents) else "",
                    metadata=metadata,
                )
            )
        return tuple(results)

    def _first_query_values(self, raw: dict[str, Any], key: str) -> list[Any]:
        value = raw.get(key) or [[]]
        if not value:
            return []
        first = value[0]
        return list(first or [])

    def _score_from_distance(self, distance: float) -> float:
        if distance < 0:
            return 0.0
        return round(1.0 / (1.0 + distance), 6)

    def _is_not_found_error(self, exc: Exception) -> bool:
        return exc.__class__.__name__ == "NotFoundError" or "does not exist" in str(exc)
