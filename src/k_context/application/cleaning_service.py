from __future__ import annotations

import re
from dataclasses import replace

from k_context.domain.models import DocumentBlock


BASIC_PROFILE = "basic"
_WHITESPACE_PATTERN = re.compile(r"\s+", re.UNICODE)


class CleaningService:
    """
    清洗 DocumentBlock 文本，同时保留可追溯字段。

    """

    def clean(
        self,
        blocks: tuple[DocumentBlock, ...],
        *,
        cleaning_profile: str = BASIC_PROFILE,
    ) -> tuple[DocumentBlock, ...]:
        if cleaning_profile != BASIC_PROFILE:
            raise ValueError(f"Unsupported cleaning_profile: {cleaning_profile}")

        cleaned_blocks: list[DocumentBlock] = []
        for block in blocks:
            cleaned_text = self._clean_text(block.text)
            if not cleaned_text:
                continue
            cleaned_blocks.append(replace(block, text=cleaned_text))
        return tuple(cleaned_blocks)

    def _clean_text(self, text: str) -> str:
        normalized_newlines = text.replace("\r\n", "\n").replace("\r", "\n")
        return _WHITESPACE_PATTERN.sub(" ", normalized_newlines).strip()
