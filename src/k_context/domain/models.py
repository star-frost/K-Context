"""已实现知识库切片使用的核心数据对象。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any
from uuid import uuid4

DEFAULT_CONFIG_VALUES: dict[str, Any] = {
    "embedding_model": "bge-m3",
    "embedding_device": "auto",
    "vector_store_type": "chroma",
    "chroma_persist_dir": "index/chroma",
    "chunking_strategy": "traditional",
    "cleaning_profile": "basic",
    "retrieval_mode": "vector",
    "rag_method": "standard",
    "top_k": 5,
    "llm_base_url": None,
    "llm_model": None,
}


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def chunk_text_hash(text: str) -> str:
    """返回用于检测 embedding 输入变化的稳定哈希。"""

    return sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Document:
    """符合文档 Document 契约的已导入本地文件元数据。"""

    doc_id: str
    file_name: str
    file_type: str
    storage_ref: str
    imported_at: str
    status: str
    error_message: str | None

    @classmethod
    def create(
        cls,
        *,
        file_name: str,
        file_type: str,
        storage_ref: str,
        status: str,
        error_message: str | None,
    ) -> "Document":
        return cls(
            doc_id=_new_id("doc"),
            file_name=file_name,
            file_type=file_type,
            storage_ref=storage_ref,
            imported_at=_utc_now(),
            status=status,
            error_message=error_message,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "file_name": self.file_name,
            "file_type": self.file_type,
            "storage_ref": self.storage_ref,
            "imported_at": self.imported_at,
            "status": self.status,
            "error_message": self.error_message,
        }


@dataclass(frozen=True)
class ParsedBlock:
    """分配来源文档标识前的解析器输出。"""

    page: int | None
    order: int
    block_type: str
    heading_path: tuple[str, ...]
    text: str
    bbox: object | None


@dataclass(frozen=True)
class DocumentBlock:
    """符合文档 DocumentBlock 契约的统一 IR 块。"""

    block_id: str
    source_doc_id: str
    source_doc_name: str
    page: int | None
    order: int
    block_type: str
    heading_path: tuple[str, ...]
    text: str
    bbox: object | None

    @classmethod
    def create(
        cls,
        *,
        source_doc_id: str,
        source_doc_name: str,
        page: int | None,
        order: int,
        block_type: str,
        heading_path: tuple[str, ...],
        text: str,
        bbox: object | None,
    ) -> "DocumentBlock":
        return cls(
            block_id=_new_id("block"),
            source_doc_id=source_doc_id,
            source_doc_name=source_doc_name,
            page=page,
            order=order,
            block_type=block_type,
            heading_path=heading_path,
            text=text,
            bbox=bbox,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "source_doc_id": self.source_doc_id,
            "source_doc_name": self.source_doc_name,
            "page": self.page,
            "order": self.order,
            "block_type": self.block_type,
            "heading_path": list(self.heading_path),
            "text": self.text,
            "bbox": self.bbox,
        }

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> "DocumentBlock":
        return cls(
            block_id=str(record["block_id"]),
            source_doc_id=str(record["source_doc_id"]),
            source_doc_name=str(record["source_doc_name"]),
            page=record["page"],
            order=int(record["order"]),
            block_type=str(record["block_type"]),
            heading_path=tuple(str(item) for item in record["heading_path"]),
            text=str(record["text"]),
            bbox=record["bbox"],
        )


@dataclass(frozen=True)
class Chunk:
    """符合文档 Chunk 契约的可检索文本单元。"""

    chunk_id: str
    source_doc_id: str
    source_doc_name: str
    page_start: int | None
    page_end: int | None
    heading_path: tuple[str, ...]
    block_ids: tuple[str, ...]
    text: str

    @classmethod
    def create(
        cls,
        *,
        source_doc_id: str,
        source_doc_name: str,
        page_start: int | None,
        page_end: int | None,
        heading_path: tuple[str, ...],
        block_ids: tuple[str, ...],
        text: str,
    ) -> "Chunk":
        return cls(
            chunk_id=_new_id("chunk"),
            source_doc_id=source_doc_id,
            source_doc_name=source_doc_name,
            page_start=page_start,
            page_end=page_end,
            heading_path=heading_path,
            block_ids=block_ids,
            text=text,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "source_doc_id": self.source_doc_id,
            "source_doc_name": self.source_doc_name,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "heading_path": list(self.heading_path),
            "block_ids": list(self.block_ids),
            "text": self.text,
        }

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> "Chunk":
        return cls(
            chunk_id=str(record["chunk_id"]),
            source_doc_id=str(record["source_doc_id"]),
            source_doc_name=str(record["source_doc_name"]),
            page_start=record["page_start"],
            page_end=record["page_end"],
            heading_path=tuple(str(item) for item in record["heading_path"]),
            block_ids=tuple(str(item) for item in record["block_ids"]),
            text=str(record["text"]),
        )


@dataclass(frozen=True)
class KContextConfig:
    """本地知识库配置的运行时视图。"""

    embedding_model: str
    embedding_device: str
    vector_store_type: str
    chroma_persist_dir: str
    chunking_strategy: str
    cleaning_profile: str
    retrieval_mode: str
    rag_method: str
    top_k: int
    llm_base_url: str | None
    llm_model: str | None

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "KContextConfig":
        merged = dict(DEFAULT_CONFIG_VALUES)
        merged.update({key: value for key, value in values.items() if value is not None})
        top_k = int(merged["top_k"])
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer.")
        return cls(
            embedding_model=str(merged["embedding_model"]),
            embedding_device=str(merged["embedding_device"]),
            vector_store_type=str(merged["vector_store_type"]),
            chroma_persist_dir=str(merged["chroma_persist_dir"]),
            chunking_strategy=str(merged["chunking_strategy"]),
            cleaning_profile=str(merged["cleaning_profile"]),
            retrieval_mode=str(merged["retrieval_mode"]),
            rag_method=str(merged["rag_method"]),
            top_k=top_k,
            llm_base_url=(
                str(merged["llm_base_url"]) if merged["llm_base_url"] is not None else None
            ),
            llm_model=str(merged["llm_model"]) if merged["llm_model"] is not None else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "embedding_model": self.embedding_model,
            "embedding_device": self.embedding_device,
            "vector_store_type": self.vector_store_type,
            "chroma_persist_dir": self.chroma_persist_dir,
            "chunking_strategy": self.chunking_strategy,
            "cleaning_profile": self.cleaning_profile,
            "retrieval_mode": self.retrieval_mode,
            "rag_method": self.rag_method,
            "top_k": self.top_k,
            "llm_base_url": self.llm_base_url,
            "llm_model": self.llm_model,
        }


@dataclass(frozen=True)
class EmbeddingRecord:
    """一个来源切块或等价来源文本的 embedding 输出。"""

    embedding_id: str
    chunk_id: str
    embedding: tuple[float, ...]
    embedding_model: str
    embedding_dim: int
    chunk_text_hash: str
    created_at: str
    status: str

    @classmethod
    def create(
        cls,
        *,
        chunk_id: str,
        text: str,
        embedding: tuple[float, ...],
        embedding_model: str,
        status: str = "created",
    ) -> "EmbeddingRecord":
        return cls(
            embedding_id=_new_id("embedding"),
            chunk_id=chunk_id,
            embedding=embedding,
            embedding_model=embedding_model,
            embedding_dim=len(embedding),
            chunk_text_hash=chunk_text_hash(text),
            created_at=_utc_now(),
            status=status,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "embedding_id": self.embedding_id,
            "chunk_id": self.chunk_id,
            "embedding": list(self.embedding),
            "embedding_model": self.embedding_model,
            "embedding_dim": self.embedding_dim,
            "chunk_text_hash": self.chunk_text_hash,
            "created_at": self.created_at,
            "status": self.status,
        }


@dataclass(frozen=True)
class QueryEmbedding:
    """一个检索查询的 embedding 输出。"""

    text: str
    embedding: tuple[float, ...]
    embedding_model: str
    embedding_dim: int
    text_hash: str
    created_at: str

    @classmethod
    def create(
        cls,
        *,
        text: str,
        embedding: tuple[float, ...],
        embedding_model: str,
    ) -> "QueryEmbedding":
        return cls(
            text=text,
            embedding=embedding,
            embedding_model=embedding_model,
            embedding_dim=len(embedding),
            text_hash=chunk_text_hash(text),
            created_at=_utc_now(),
        )


@dataclass(frozen=True)
class VectorRecord:
    """将一个切块 embedding 与可追溯元数据绑定的向量存储输入。"""

    chunk_id: str
    embedding: tuple[float, ...]
    source_doc_id: str
    source_doc_name: str
    block_ids: tuple[str, ...]
    text: str
    chunking_strategy: str
    cleaning_profile: str
    embedding_model: str
    embedding_dim: int
    chunk_text_hash: str
    index_version: str
    created_at: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        chunk: Chunk,
        embedding_record: EmbeddingRecord,
        chunking_strategy: str,
        cleaning_profile: str,
        index_version: str,
        metadata: dict[str, Any] | None = None,
    ) -> "VectorRecord":
        if chunk.chunk_id != embedding_record.chunk_id:
            raise ValueError("VectorRecord chunk_id must match embedding_record chunk_id.")
        if len(embedding_record.embedding) != embedding_record.embedding_dim:
            raise ValueError("Embedding vector length must match embedding_dim.")
        return cls(
            chunk_id=chunk.chunk_id,
            embedding=embedding_record.embedding,
            source_doc_id=chunk.source_doc_id,
            source_doc_name=chunk.source_doc_name,
            block_ids=chunk.block_ids,
            text=chunk.text,
            chunking_strategy=chunking_strategy,
            cleaning_profile=cleaning_profile,
            embedding_model=embedding_record.embedding_model,
            embedding_dim=embedding_record.embedding_dim,
            chunk_text_hash=embedding_record.chunk_text_hash,
            index_version=index_version,
            created_at=_utc_now(),
            metadata=dict(metadata or {}),
        )

    def search_metadata(self) -> dict[str, Any]:
        payload = {
            "chunk_id": self.chunk_id,
            "source_doc_id": self.source_doc_id,
            "source_doc_name": self.source_doc_name,
            "block_ids": list(self.block_ids),
            "chunking_strategy": self.chunking_strategy,
            "cleaning_profile": self.cleaning_profile,
            "embedding_model": self.embedding_model,
            "embedding_dim": self.embedding_dim,
            "chunk_text_hash": self.chunk_text_hash,
            "index_version": self.index_version,
            "created_at": self.created_at,
        }
        payload.update(self.metadata)
        return payload

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "embedding": list(self.embedding),
            "source_doc_id": self.source_doc_id,
            "source_doc_name": self.source_doc_name,
            "block_ids": list(self.block_ids),
            "text": self.text,
            "chunking_strategy": self.chunking_strategy,
            "cleaning_profile": self.cleaning_profile,
            "embedding_model": self.embedding_model,
            "embedding_dim": self.embedding_dim,
            "chunk_text_hash": self.chunk_text_hash,
            "index_version": self.index_version,
            "created_at": self.created_at,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class VectorSearchResult:
    """与文档 SearchResult 字段兼容的向量检索结果。"""

    chunk_id: str
    source_doc_id: str
    source_doc_name: str
    score: float
    retrieval_mode: str
    block_ids: tuple[str, ...]
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_vector_record(
        cls,
        *,
        record: VectorRecord,
        score: float,
        retrieval_mode: str = "vector",
    ) -> "VectorSearchResult":
        return cls(
            chunk_id=record.chunk_id,
            source_doc_id=record.source_doc_id,
            source_doc_name=record.source_doc_name,
            score=score,
            retrieval_mode=retrieval_mode,
            block_ids=record.block_ids,
            text=record.text,
            metadata=record.search_metadata(),
        )

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
