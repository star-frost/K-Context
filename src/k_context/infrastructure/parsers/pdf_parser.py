"""Minimal ordinary-PDF text parser using only the Python standard library."""

from __future__ import annotations

import re
import zlib
from pathlib import Path

from k_context.domain.models import ParsedBlock


STREAM_PATTERN = re.compile(rb"(<<.*?>>)\s*stream\r?\n(.*?)\r?\nendstream", re.DOTALL)
STRING_PATTERN = re.compile(r"\((?:\\.|[^\\)])*\)")


def parse_pdf(file_path: Path) -> tuple[ParsedBlock, ...]:
    data = file_path.read_bytes()
    if not data.startswith(b"%PDF"):
        return ()

    texts: list[str] = []
    for stream_match in STREAM_PATTERN.finditer(data):
        stream_dict = stream_match.group(1)
        stream_data = stream_match.group(2)
        if b"/FlateDecode" in stream_dict:
            try:
                stream_data = zlib.decompress(stream_data)
            except zlib.error:
                continue
        decoded = stream_data.decode("latin-1", errors="ignore")
        texts.extend(_extract_literal_strings(decoded))

    joined = " ".join(text for text in texts if text.strip()).strip()
    if not joined:
        return ()

    return (
        ParsedBlock(
            page=1,
            order=0,
            block_type="paragraph",
            heading_path=(),
            text=joined,
            bbox=None,
        ),
    )


def _extract_literal_strings(content: str) -> list[str]:
    return [_decode_pdf_literal(match.group(0)[1:-1]) for match in STRING_PATTERN.finditer(content)]


def _decode_pdf_literal(value: str) -> str:
    result: list[str] = []
    index = 0
    while index < len(value):
        char = value[index]
        if char != "\\":
            result.append(char)
            index += 1
            continue

        index += 1
        if index >= len(value):
            break
        escaped = value[index]
        replacements = {"n": "\n", "r": "\r", "t": "\t", "b": "\b", "f": "\f"}
        result.append(replacements.get(escaped, escaped))
        index += 1
    return "".join(result).strip()
