"""Embedding provider 契约。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from k_context.domain.models import EmbeddingRecord, QueryEmbedding


DEFAULT_BGE_MODEL = "bge-m3"
SUPPORTED_EMBEDDING_DEVICES = {"auto", "cpu", "cuda"}


class EmbeddingProviderError(RuntimeError):
    """当真实 embedding provider 无法加载或执行时抛出。"""


@dataclass(frozen=True)
class EmbeddingInput:
    """用于构建 EmbeddingRecord 的文本和来源标识。"""

    source_id: str
    text: str


class EmbeddingProvider(Protocol):
    """文档与查询 embedding 生成边界。"""

    embedding_model: str
    embedding_dim: int

    def embed_documents(self, inputs: tuple[EmbeddingInput, ...]) -> tuple[EmbeddingRecord, ...]:
        """对切块文本或等价来源文本生成 embedding。"""

    def embed_query(self, text: str) -> QueryEmbedding:
        """对一个检索查询生成 embedding。"""
