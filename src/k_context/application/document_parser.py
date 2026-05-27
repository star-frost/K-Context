"""用于最小 add 流程的文档解析应用服务。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from k_context.domain.models import ParsedBlock
from k_context.infrastructure.parsers.markdown_parser import parse_markdown
from k_context.infrastructure.parsers.pdf_parser import parse_pdf
from k_context.infrastructure.parsers.text_parser import parse_text


SUPPORTED_TYPES = {
    ".pdf": "pdf",
    ".md": "md",
    ".markdown": "md",
    ".txt": "txt",
}


class DocumentParseError(ValueError):
    """当文件无法解析为非空文本块时抛出。"""


@dataclass(frozen=True)
class ParsedDocument:
    """已解析文档文本和规范化文件类型。"""

    file_path: Path
    file_type: str
    blocks: tuple[ParsedBlock, ...]


class DocumentParser:
    """将受支持的本地文件转换为可生成 DocumentBlock 的最小文本块。"""

    def parse(self, file_path: Path) -> ParsedDocument:
        resolved_path = file_path.expanduser().resolve()
        if not resolved_path.is_file():
            raise FileNotFoundError(f"File does not exist: {file_path}")

        file_type = SUPPORTED_TYPES.get(resolved_path.suffix.lower())
        if file_type is None:
            raise DocumentParseError(
                "Unsupported file type. Only PDF, Markdown, and TXT files are supported."
            )

        blocks = self._parse_supported_file(resolved_path, file_type)
        if not blocks:
            raise DocumentParseError("No extractable text was found in the document.")

        return ParsedDocument(file_path=resolved_path, file_type=file_type, blocks=tuple(blocks))

    def _parse_supported_file(self, file_path: Path, file_type: str) -> tuple[ParsedBlock, ...]:
        if file_type == "txt":
            return parse_text(file_path)
        if file_type == "md":
            return parse_markdown(file_path)
        if file_type == "pdf":
            return parse_pdf(file_path)
        raise DocumentParseError(f"Unsupported file type: {file_type}")
