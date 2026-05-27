"""由 docling 支撑、带最小旧版兜底的 PDF 解析器适配器。"""

from __future__ import annotations

import re
import zlib
from pathlib import Path
from typing import Any

from k_context.domain.models import ParsedBlock


STREAM_PATTERN = re.compile(rb"(<<.*?>>)\s*stream\r?\n(.*?)\r?\nendstream", re.DOTALL)
STRING_PATTERN = re.compile(r"\((?:\\.|[^\\)])*\)")
HEADING_LABELS = {"section_header", "title"}


def parse_pdf(file_path: Path) -> tuple[ParsedBlock, ...]:
    """将普通 PDF 解析为多个文本块。

    Docling 是主解析器，因为真实 PDF 经常使用编码字体、
    压缩内容流以及布局结构，旧的字面字符串抽取器无法可靠解码。
    仅保留一个很小的旧版兜底，用于超小 fixture PDF 或 docling
    无法解析文件的环境。
    """

    data = file_path.read_bytes()
    if not data.startswith(b"%PDF"):
        return ()

    blocks = _parse_pdf_with_docling(file_path)
    if blocks and _text_quality_is_usable(blocks):
        return blocks
    blocks = _parse_pdf_with_pypdf(file_path)
    if blocks:
        return blocks
    return _parse_pdf_legacy_literal_streams(data)


def _parse_pdf_with_docling(file_path: Path) -> tuple[ParsedBlock, ...]:
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
    except ImportError:
        return ()

    try:
        pipeline_options = PdfPipelineOptions()
        # OCR 属于独立增强路径。对于原生数字 PDF，优先使用后端文本抽取，
        # 避免 OCR 下载和噪声。
        pipeline_options.do_ocr = False
        pipeline_options.do_table_structure = False
        converter = DocumentConverter(
            allowed_formats=[InputFormat.PDF],
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
            },
        )
        result = converter.convert(file_path)
    except Exception:
        return ()

    document = getattr(result, "document", None)
    if document is None:
        return ()
    return _blocks_from_docling_document(document)


def _blocks_from_docling_document(document: Any) -> tuple[ParsedBlock, ...]:
    heading_stack: list[str] = []
    blocks: list[ParsedBlock] = []

    for item in getattr(document, "texts", ()) or ():
        text = _normalize_text(str(getattr(item, "text", "") or ""))
        if not text:
            continue

        label = _label_value(getattr(item, "label", ""))
        if label in HEADING_LABELS:
            level = max(1, int(getattr(item, "level", 1) or 1))
            heading_stack = heading_stack[: level - 1]
            heading_stack.append(text)
            block_type = "title"
            heading_path = tuple(heading_stack)
        else:
            block_type = "paragraph"
            heading_path = tuple(heading_stack)

        blocks.append(
            ParsedBlock(
                page=_first_page(item),
                order=len(blocks),
                block_type=block_type,
                heading_path=heading_path,
                text=text,
                bbox=_first_bbox(item),
            )
        )

    return tuple(blocks)


def _parse_pdf_with_pypdf(file_path: Path) -> tuple[ParsedBlock, ...]:
    try:
        from pypdf import PdfReader
    except ImportError:
        return ()

    try:
        reader = PdfReader(str(file_path))
    except Exception:
        return ()

    blocks: list[ParsedBlock] = []
    for page_index, page in enumerate(reader.pages, start=1):
        try:
            text = _normalize_text(page.extract_text() or "")
        except Exception:
            continue
        if not text:
            continue
        blocks.append(
            ParsedBlock(
                page=page_index,
                order=len(blocks),
                block_type="paragraph",
                heading_path=(),
                text=text,
                bbox=None,
            )
        )
    return tuple(blocks)


def _text_quality_is_usable(blocks: tuple[ParsedBlock, ...]) -> bool:
    text = "\n".join(block.text for block in blocks)
    if not text.strip():
        return False
    if len(blocks) <= 2 and len(text) > 100_000:
        return False
    sample = text[:20000]
    control_chars = sum(
        1
        for char in sample
        if ord(char) < 32 and char not in {"\n", "\t", "\r"}
    )
    if control_chars / max(1, len(sample)) > 0.05:
        return False
    printable = sum(1 for char in sample if char.isprintable() or char in {"\n", "\t", "\r"})
    return printable / max(1, len(sample)) > 0.9


def _label_value(label: Any) -> str:
    return str(getattr(label, "value", label)).strip().lower()


def _first_page(item: Any) -> int | None:
    prov = getattr(item, "prov", None) or ()
    if not prov:
        return None
    page_no = getattr(prov[0], "page_no", None)
    return int(page_no) if page_no is not None else None


def _first_bbox(item: Any) -> dict[str, object] | None:
    prov = getattr(item, "prov", None) or ()
    if not prov:
        return None
    bbox = getattr(prov[0], "bbox", None)
    if bbox is None:
        return None
    raw = bbox.model_dump() if hasattr(bbox, "model_dump") else dict(bbox)
    return {str(key): _json_safe(value) for key, value in raw.items()}


def _json_safe(value: Any) -> object:
    if isinstance(value, int | float | str | bool) or value is None:
        return value
    if hasattr(value, "value"):
        return str(value.value)
    return str(value)


def _normalize_text(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text.replace("\r\n", "\n").replace("\r", "\n")).strip()


def _parse_pdf_legacy_literal_streams(data: bytes) -> tuple[ParsedBlock, ...]:
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
