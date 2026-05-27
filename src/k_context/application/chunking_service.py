from __future__ import annotations

import re
from typing import Protocol

from k_context.application.embedding_provider import EmbeddingInput, EmbeddingProvider
from k_context.domain.models import Chunk, DocumentBlock


TARGET_CHUNK_MAX = 1000
HARD_CHUNK_MAX = 1500
DSC_SOFT_MIN = 350
DSC_SIMILARITY_THRESHOLD = 0.12
DSC_EMBEDDING_SIMILARITY_THRESHOLD = 0.55
TRADITIONAL_CHUNKING_STRATEGY = "traditional"
DSC_CHUNKING_STRATEGY = "dsc"
_TOKEN_PATTERN = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)
_STOPWORDS = {
    "the",
    "and",
    "for",
    "that",
    "with",
    "from",
    "this",
    "these",
    "those",
    "are",
    "was",
    "were",
    "into",
    "their",
    "there",
    "which",
    "using",
    "used",
    "use",
    "can",
    "will",
    "may",
    "our",
    "between",
    "without",
    "within",
}


class ChunkingStrategy(Protocol):
    """将清洗后块转换为可检索切块的策略契约。"""

    def generate(self, blocks: tuple[DocumentBlock, ...]) -> tuple[Chunk, ...]:
        """从清洗后的 DocumentBlock 记录生成切块。"""


class TraditionalChunkingStrategy:
    """当前基线切块行为，保留为 traditional 策略。"""

    name = TRADITIONAL_CHUNKING_STRATEGY

    def generate(self, blocks: tuple[DocumentBlock, ...]) -> tuple[Chunk, ...]:
        valid_blocks = tuple(
            block for block in sorted(blocks, key=lambda item: (item.source_doc_id, item.order))
            if block.text.strip()
        )
        if not valid_blocks:
            return ()

        chunks: list[Chunk] = []
        current_group: list[DocumentBlock] = []

        for block in valid_blocks:
            split_texts = self._split_text(block.text.strip())
            if len(split_texts) > 1:
                if current_group:
                    chunks.append(self._build_chunk(current_group))
                    current_group = []
                for text_part in split_texts:
                    chunks.append(self._build_chunk((block,), override_text=text_part))
                continue

            if self._should_start_new_chunk(current_group, block):
                chunks.append(self._build_chunk(current_group))
                current_group = []

            current_group.append(block)

        if current_group:
            chunks.append(self._build_chunk(current_group))

        return tuple(chunks)

    def _should_start_new_chunk(self, current_group: list[DocumentBlock], block: DocumentBlock) -> bool:
        if not current_group:
            return False
        if current_group[-1].source_doc_id != block.source_doc_id:
            return True
        combined_text = self._join_block_texts((*current_group, block))
        return len(combined_text) > TARGET_CHUNK_MAX

    def _build_chunk(
        self,
        blocks: tuple[DocumentBlock, ...] | list[DocumentBlock],
        *,
        override_text: str | None = None,
    ) -> Chunk:
        pages = [block.page for block in blocks if block.page is not None]
        heading_path = self._select_heading_path(blocks)
        return Chunk.create(
            source_doc_id=blocks[0].source_doc_id,
            source_doc_name=blocks[0].source_doc_name,
            page_start=min(pages) if pages else None,
            page_end=max(pages) if pages else None,
            heading_path=heading_path,
            block_ids=tuple(block.block_id for block in blocks),
            text=override_text or self._join_block_texts(blocks),
        )

    def _select_heading_path(self, blocks: tuple[DocumentBlock, ...] | list[DocumentBlock]) -> tuple[str, ...]:
        for block in blocks:
            if block.heading_path:
                return block.heading_path
        return ()

    def _join_block_texts(self, blocks: tuple[DocumentBlock, ...] | list[DocumentBlock]) -> str:
        return "\n\n".join(block.text.strip() for block in blocks if block.text.strip())

    def _split_text(self, text: str) -> tuple[str, ...]:
        if len(text) <= HARD_CHUNK_MAX:
            return (text,)
        return tuple(
            text[index : index + HARD_CHUNK_MAX].strip()
            for index in range(0, len(text), HARD_CHUNK_MAX)
            if text[index : index + HARD_CHUNK_MAX].strip()
        )


class DualSemanticChunkingStrategy(TraditionalChunkingStrategy):

    name = DSC_CHUNKING_STRATEGY

    def __init__(
        self,
        *,
        embedding_provider: EmbeddingProvider | None = None,
        embedding_similarity_threshold: float = DSC_EMBEDDING_SIMILARITY_THRESHOLD,
    ) -> None:
        self._embedding_provider = embedding_provider
        self._embedding_similarity_threshold = embedding_similarity_threshold
        self._embedding_vectors_by_block_id: dict[str, tuple[float, ...]] = {}

    def with_embedding_provider(
        self,
        embedding_provider: EmbeddingProvider,
    ) -> "DualSemanticChunkingStrategy":
        return DualSemanticChunkingStrategy(
            embedding_provider=embedding_provider,
            embedding_similarity_threshold=self._embedding_similarity_threshold,
        )

    def generate(self, blocks: tuple[DocumentBlock, ...]) -> tuple[Chunk, ...]:
        valid_blocks = tuple(
            block for block in sorted(blocks, key=lambda item: (item.source_doc_id, item.order))
            if block.text.strip()
        )
        if not valid_blocks:
            return ()

        self._embedding_vectors_by_block_id = self._embed_blocks(valid_blocks)
        try:
            chunks: list[Chunk] = []
            current_group: list[DocumentBlock] = []

            for block in valid_blocks:
                split_texts = self._split_text(block.text.strip())
                if len(split_texts) > 1:
                    if current_group:
                        chunks.append(self._build_chunk(current_group))
                        current_group = []
                    for text_part in split_texts:
                        chunks.append(self._build_chunk((block,), override_text=text_part))
                    continue

                if self._should_start_new_chunk(current_group, block):
                    chunks.append(self._build_chunk(current_group))
                    current_group = []

                current_group.append(block)

            if current_group:
                chunks.append(self._build_chunk(current_group))

            return tuple(chunks)
        finally:
            self._embedding_vectors_by_block_id = {}

    def _embed_blocks(self, blocks: tuple[DocumentBlock, ...]) -> dict[str, tuple[float, ...]]:
        if self._embedding_provider is None:
            return {}
        records = self._embedding_provider.embed_documents(
            tuple(
                EmbeddingInput(source_id=block.block_id, text=block.text.strip())
                for block in blocks
            )
        )
        return {record.chunk_id: tuple(record.embedding) for record in records}

    def _should_start_new_chunk(self, current_group: list[DocumentBlock], block: DocumentBlock) -> bool:
        if not current_group:
            return False
        if current_group[-1].source_doc_id != block.source_doc_id:
            return True

        current_text = self._join_block_texts(current_group)
        combined_text = self._join_block_texts((*current_group, block))
        if len(combined_text) > HARD_CHUNK_MAX:
            return True
        if len(combined_text) > TARGET_CHUNK_MAX and len(current_text) >= DSC_SOFT_MIN:
            return True

        if len(current_text) < DSC_SOFT_MIN:
            return False

        if self._is_structural_boundary(current_group[-1], block):
            return True

        similarity = self._semantic_similarity(current_group, block)
        threshold = (
            self._embedding_similarity_threshold
            if self._embedding_vectors_by_block_id
            else DSC_SIMILARITY_THRESHOLD
        )
        return similarity < threshold

    def _is_structural_boundary(self, previous: DocumentBlock, block: DocumentBlock) -> bool:
        block_type = (block.block_type or "").lower()
        if block_type in {"heading", "header", "section_header", "title"}:
            return True

        previous_path = tuple(previous.heading_path or ())
        next_path = tuple(block.heading_path or ())
        return bool(previous_path and next_path and previous_path != next_path)

    def _semantic_similarity(self, current_group: list[DocumentBlock], block: DocumentBlock) -> float:
        embedding_similarity = self._embedding_similarity(current_group, block)
        if embedding_similarity is not None:
            return embedding_similarity

        left = _semantic_tokens("\n".join(item.text for item in current_group[-2:]))
        right = _semantic_tokens(block.text)
        if not left or not right:
            return 0.0
        return len(left & right) / len(left | right)

    def _embedding_similarity(self, current_group: list[DocumentBlock], block: DocumentBlock) -> float | None:
        if not self._embedding_vectors_by_block_id:
            return None
        left_vectors = tuple(
            vector
            for item in current_group[-2:]
            if (vector := self._embedding_vectors_by_block_id.get(item.block_id)) is not None
        )
        right_vector = self._embedding_vectors_by_block_id.get(block.block_id)
        if not left_vectors or right_vector is None:
            return None
        return _cosine_similarity(_mean_vector(left_vectors), right_vector)


def _semantic_tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for match in _TOKEN_PATTERN.finditer(text.lower()):
        token = match.group(0).strip("_")
        if len(token) < 3 or token in _STOPWORDS:
            continue
        tokens.add(token)
    return tokens


def _mean_vector(vectors: tuple[tuple[float, ...], ...]) -> tuple[float, ...]:
    if not vectors:
        return ()
    dimension = len(vectors[0])
    if dimension == 0:
        return ()
    return tuple(sum(vector[index] for vector in vectors) / len(vectors) for index in range(dimension))


def _cosine_similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(left_value * right_value for left_value, right_value in zip(left, right, strict=True))
    left_norm = sum(value * value for value in left) ** 0.5
    right_norm = sum(value * value for value in right) ** 0.5
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


class ChunkingService:
    """
    选择切块策略并生成切块。

    """

    def __init__(
        self,
        strategies: tuple[ChunkingStrategy, ...] | None = None,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        registered_strategies = strategies or (
            TraditionalChunkingStrategy(),
            DualSemanticChunkingStrategy(embedding_provider=embedding_provider),
        )
        self._strategies = {
            getattr(strategy, "name", strategy.__class__.__name__): strategy
            for strategy in registered_strategies
        }

    def generate(
        self,
        blocks: tuple[DocumentBlock, ...],
        *,
        chunking_strategy: str = TRADITIONAL_CHUNKING_STRATEGY,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> tuple[Chunk, ...]:
        try:
            strategy = self._strategies[chunking_strategy]
        except KeyError as exc:
            supported = ", ".join(sorted(self._strategies))
            raise ValueError(
                f"Unsupported chunking_strategy: {chunking_strategy}. Supported strategies: {supported}"
            ) from exc
        if embedding_provider is not None and isinstance(strategy, DualSemanticChunkingStrategy):
            strategy = strategy.with_embedding_provider(embedding_provider)
        return strategy.generate(blocks)
