"""Application service for local knowledge-base lifecycle operations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from k_context.application.answer_service import GroundedAnswer, GroundedAnswerService
from k_context.application.chunking_service import ChunkingService
from k_context.application.document_parser import DocumentParser
from k_context.application.retrieval_service import DEFAULT_TOP_K, RetrievalService, SearchResult
from k_context.domain.models import Chunk, Document, DocumentBlock, ParsedBlock
from k_context.infrastructure.storage.local_store import KbInitRecord, LocalKnowledgeBaseStore


@dataclass(frozen=True)
class KbInitResult:
    """Result returned after ensuring a local knowledge base exists."""

    kb_root: Path
    created_paths: tuple[Path, ...]
    already_initialized: bool


@dataclass(frozen=True)
class KbAddResult:
    """Result returned after adding and parsing one local document."""

    document: Document
    blocks: tuple[DocumentBlock, ...]
    metadata_path: Path
    blocks_path: Path


@dataclass(frozen=True)
class ChunkingResult:
    """Result returned after generating chunks from persisted DocumentBlock records."""

    chunks: tuple[Chunk, ...]
    chunks_path: Path


@dataclass(frozen=True)
class SearchResults:
    """Result returned after searching persisted chunks."""

    results: tuple[SearchResult, ...]
    chunks_path: Path
    chunks_available: int


class KnowledgeBaseService:
    """Coordinates knowledge-base operations without exposing storage details to CLI."""

    def __init__(
        self,
        store: LocalKnowledgeBaseStore | None = None,
        parser: DocumentParser | None = None,
        chunking: ChunkingService | None = None,
        retrieval: RetrievalService | None = None,
        answer_service: GroundedAnswerService | None = None,
    ) -> None:
        self._store = store or LocalKnowledgeBaseStore()
        self._parser = parser or DocumentParser()
        self._chunking = chunking or ChunkingService()
        self._retrieval = retrieval or RetrievalService()
        self._answer_service = answer_service or GroundedAnswerService()

    def init(self, root: Path) -> KbInitResult:
        """Create the local knowledge-base directory layout under ``root``."""

        record: KbInitRecord = self._store.initialize(root)
        return KbInitResult(
            kb_root=record.kb_root,
            created_paths=record.created_paths,
            already_initialized=record.already_initialized,
        )

    def add(self, root: Path, file_path: Path) -> KbAddResult:
        """Register a document, parse it to IR, and persist metadata and blocks."""

        kb_paths = self._store.require_initialized(root)
        parsed_document = self._parser.parse(file_path)
        document = Document.create(
            file_name=parsed_document.file_path.name,
            file_type=parsed_document.file_type,
            storage_ref=str(parsed_document.file_path),
            status="已解析",
            error_message=None,
        )
        blocks = tuple(
            self._to_document_block(document=document, parsed_block=block)
            for block in parsed_document.blocks
        )

        self._store.append_record(kb_paths.metadata_path, document.to_dict())
        self._store.append_records(kb_paths.blocks_path, (block.to_dict() for block in blocks))

        return KbAddResult(
            document=document,
            blocks=blocks,
            metadata_path=kb_paths.metadata_path,
            blocks_path=kb_paths.blocks_path,
        )

    def generate_chunks(self, root: Path) -> ChunkingResult:
        """Generate chunks from persisted DocumentBlock IR and replace chunks.jsonl."""

        kb_paths = self._store.require_initialized(root)
        block_records = self._store.read_records(kb_paths.blocks_path)
        blocks = tuple(DocumentBlock.from_dict(record) for record in block_records)
        chunks = self._chunking.generate(blocks)
        self._store.replace_records(kb_paths.chunks_path, (chunk.to_dict() for chunk in chunks))
        return ChunkingResult(chunks=chunks, chunks_path=kb_paths.chunks_path)

    def search(self, root: Path, query: str, top_k: int = DEFAULT_TOP_K) -> SearchResults:
        """Search persisted chunks with a lightweight local retrieval strategy."""

        kb_paths = self._store.require_initialized(root)
        chunk_records = self._store.read_records(kb_paths.chunks_path)
        chunks = tuple(Chunk.from_dict(record) for record in chunk_records)
        results = self._retrieval.search(chunks=chunks, query=query, top_k=top_k)
        return SearchResults(
            results=results,
            chunks_path=kb_paths.chunks_path,
            chunks_available=len(chunks),
        )

    def ask(self, root: Path, question: str, top_k: int = DEFAULT_TOP_K) -> GroundedAnswer:
        """Answer a question using retrieved chunks only."""

        search_results = self.search(root=root, query=question, top_k=top_k)
        return self._answer_service.synthesize(question, search_results.results)

    def _to_document_block(self, document: Document, parsed_block: ParsedBlock) -> DocumentBlock:
        return DocumentBlock.create(
            source_doc_id=document.doc_id,
            source_doc_name=document.file_name,
            page=parsed_block.page,
            order=parsed_block.order,
            block_type=parsed_block.block_type,
            heading_path=parsed_block.heading_path,
            text=parsed_block.text,
            bbox=parsed_block.bbox,
        )
