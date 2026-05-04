"""Core data objects used by the implemented knowledge-base slice."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class Document:
    """Imported local file metadata matching the documented Document contract."""

    doc_id: str
    file_name: str
    file_type: str
    storage_ref: str
    imported_at: str
    status: str
    error_message: str | None

    @classmethod
    def create(
        cls,
        *,
        file_name: str,
        file_type: str,
        storage_ref: str,
        status: str,
        error_message: str | None,
    ) -> "Document":
        return cls(
            doc_id=_new_id("doc"),
            file_name=file_name,
            file_type=file_type,
            storage_ref=storage_ref,
            imported_at=_utc_now(),
            status=status,
            error_message=error_message,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "file_name": self.file_name,
            "file_type": self.file_type,
            "storage_ref": self.storage_ref,
            "imported_at": self.imported_at,
            "status": self.status,
            "error_message": self.error_message,
        }


@dataclass(frozen=True)
class ParsedBlock:
    """Parser output before source document identifiers are assigned."""

    page: int | None
    order: int
    block_type: str
    heading_path: tuple[str, ...]
    text: str
    bbox: object | None


@dataclass(frozen=True)
class DocumentBlock:
    """Unified IR block matching the documented DocumentBlock contract."""

    block_id: str
    source_doc_id: str
    source_doc_name: str
    page: int | None
    order: int
    block_type: str
    heading_path: tuple[str, ...]
    text: str
    bbox: object | None

    @classmethod
    def create(
        cls,
        *,
        source_doc_id: str,
        source_doc_name: str,
        page: int | None,
        order: int,
        block_type: str,
        heading_path: tuple[str, ...],
        text: str,
        bbox: object | None,
    ) -> "DocumentBlock":
        return cls(
            block_id=_new_id("block"),
            source_doc_id=source_doc_id,
            source_doc_name=source_doc_name,
            page=page,
            order=order,
            block_type=block_type,
            heading_path=heading_path,
            text=text,
            bbox=bbox,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "source_doc_id": self.source_doc_id,
            "source_doc_name": self.source_doc_name,
            "page": self.page,
            "order": self.order,
            "block_type": self.block_type,
            "heading_path": list(self.heading_path),
            "text": self.text,
            "bbox": self.bbox,
        }

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> "DocumentBlock":
        return cls(
            block_id=str(record["block_id"]),
            source_doc_id=str(record["source_doc_id"]),
            source_doc_name=str(record["source_doc_name"]),
            page=record["page"],
            order=int(record["order"]),
            block_type=str(record["block_type"]),
            heading_path=tuple(str(item) for item in record["heading_path"]),
            text=str(record["text"]),
            bbox=record["bbox"],
        )


@dataclass(frozen=True)
class Chunk:
    """Retrievable text unit matching the documented Chunk contract."""

    chunk_id: str
    source_doc_id: str
    source_doc_name: str
    page_start: int | None
    page_end: int | None
    heading_path: tuple[str, ...]
    block_ids: tuple[str, ...]
    text: str

    @classmethod
    def create(
        cls,
        *,
        source_doc_id: str,
        source_doc_name: str,
        page_start: int | None,
        page_end: int | None,
        heading_path: tuple[str, ...],
        block_ids: tuple[str, ...],
        text: str,
    ) -> "Chunk":
        return cls(
            chunk_id=_new_id("chunk"),
            source_doc_id=source_doc_id,
            source_doc_name=source_doc_name,
            page_start=page_start,
            page_end=page_end,
            heading_path=heading_path,
            block_ids=block_ids,
            text=text,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "source_doc_id": self.source_doc_id,
            "source_doc_name": self.source_doc_name,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "heading_path": list(self.heading_path),
            "block_ids": list(self.block_ids),
            "text": self.text,
        }

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> "Chunk":
        return cls(
            chunk_id=str(record["chunk_id"]),
            source_doc_id=str(record["source_doc_id"]),
            source_doc_name=str(record["source_doc_name"]),
            page_start=record["page_start"],
            page_end=record["page_end"],
            heading_path=tuple(str(item) for item in record["heading_path"]),
            block_ids=tuple(str(item) for item in record["block_ids"]),
            text=str(record["text"]),
        )
