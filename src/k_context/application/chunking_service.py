"""Application-layer chunk generation from DocumentBlock IR."""

from __future__ import annotations

from k_context.domain.models import Chunk, DocumentBlock


TARGET_CHUNK_MAX = 1000
HARD_CHUNK_MAX = 1500


class ChunkingService:
    """Generates retrievable chunks from cleaned DocumentBlock records."""

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
