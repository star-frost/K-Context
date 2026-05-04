"""TXT parser adapter."""

from __future__ import annotations

import re
from pathlib import Path

from k_context.domain.models import ParsedBlock


def parse_text(file_path: Path) -> tuple[ParsedBlock, ...]:
    text = file_path.read_text(encoding="utf-8-sig")
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    return tuple(
        ParsedBlock(
            page=None,
            order=index,
            block_type="paragraph",
            heading_path=(),
            text=_normalize_whitespace(paragraph),
            bbox=None,
        )
        for index, paragraph in enumerate(paragraphs)
    )


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text.replace("\r\n", "\n").replace("\r", "\n")).strip()
