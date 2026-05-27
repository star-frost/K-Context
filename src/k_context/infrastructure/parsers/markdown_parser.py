"""保留最小标题路径的 Markdown 解析器适配器。"""

from __future__ import annotations

import re
from pathlib import Path

from k_context.domain.models import ParsedBlock


HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def parse_markdown(file_path: Path) -> tuple[ParsedBlock, ...]:
    lines = file_path.read_text(encoding="utf-8-sig").splitlines()
    heading_stack: list[str] = []
    blocks: list[ParsedBlock] = []
    paragraph_lines: list[str] = []

    def flush_paragraph() -> None:
        if not paragraph_lines:
            return
        text = " ".join(line.strip() for line in paragraph_lines if line.strip()).strip()
        paragraph_lines.clear()
        if text:
            blocks.append(
                ParsedBlock(
                    page=None,
                    order=len(blocks),
                    block_type="paragraph",
                    heading_path=tuple(heading_stack),
                    text=text,
                    bbox=None,
                )
            )

    for line in lines:
        match = HEADING_PATTERN.match(line)
        if match:
            flush_paragraph()
            level = len(match.group(1))
            title = match.group(2).strip()
            heading_stack = heading_stack[: level - 1]
            heading_stack.append(title)
            blocks.append(
                ParsedBlock(
                    page=None,
                    order=len(blocks),
                    block_type="title",
                    heading_path=tuple(heading_stack),
                    text=title,
                    bbox=None,
                )
            )
            continue

        if line.strip():
            paragraph_lines.append(line)
        else:
            flush_paragraph()

    flush_paragraph()
    return tuple(blocks)
