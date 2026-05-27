"""向量存储契约。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from k_context.domain.models import QueryEmbedding, VectorRecord, VectorSearchResult


VECTOR_RETRIEVAL_MODE = "vector"


class VectorStoreError(RuntimeError):
    """当向量存储操作无法完成时抛出。"""


class VectorStoreDimensionError(VectorStoreError):
    """当已存储向量与查询向量维度不兼容时抛出。"""


@dataclass(frozen=True)
class VectorStoreUpsertResult:
    """向量存储 upsert 操作的结果摘要。"""

    collection_name: str
    upserted_count: int
    replaced_chunk_ids: tuple[str, ...]


class VectorStore(Protocol):
    """向量存储写入、查询和集合重置操作的边界。"""

    collection_name: str

    def upsert(self, records: tuple[VectorRecord, ...]) -> VectorStoreUpsertResult:
        """在当前集合中插入或替换向量记录。"""

    def query(self, query_embedding: QueryEmbedding, *, top_k: int) -> tuple[VectorSearchResult, ...]:
        """返回查询 embedding 的 top_k 向量候选。"""

    def delete_collection(self) -> None:
        """删除或清空当前集合。"""

    def rebuild_collection(self, records: tuple[VectorRecord, ...] = ()) -> VectorStoreUpsertResult:
        """清空当前集合，并可选地 upsert 替换记录。"""
