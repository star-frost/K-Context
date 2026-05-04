from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from k_context.application.kb_service import KnowledgeBaseService
from k_context.domain.models import DocumentBlock
from k_context.infrastructure.storage.local_store import LocalKnowledgeBaseStore


class ChunkingTests(unittest.TestCase):
    def test_generate_chunks_from_blocks_jsonl(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "notes.txt"
            source.write_text("Alpha text.\n\nBeta text.", encoding="utf-8")
            service = KnowledgeBaseService()

            service.init(root)
            add_result = service.add(root, source)
            chunking_result = service.generate_chunks(root)

            chunk_records = _read_jsonl(root / ".kcontext" / "chunks.jsonl")
            self.assertEqual(len(chunking_result.chunks), 1)
            self.assertEqual(chunk_records, [chunking_result.chunks[0].to_dict()])
            self.assertEqual(chunking_result.chunks_path, root / ".kcontext" / "chunks.jsonl")
            self.assertEqual(chunk_records[0]["source_doc_id"], add_result.document.doc_id)
            self.assertEqual(chunk_records[0]["source_doc_name"], "notes.txt")
            self.assertEqual(
                chunk_records[0]["block_ids"],
                [block.block_id for block in add_result.blocks],
            )

    def test_chunk_fields_match_data_contract(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            service = KnowledgeBaseService()
            service.init(root)
            _write_blocks(
                root,
                [
                    DocumentBlock.create(
                        source_doc_id="doc_1",
                        source_doc_name="paper.md",
                        page=None,
                        order=0,
                        block_type="paragraph",
                        heading_path=("Intro",),
                        text="Contract text.",
                        bbox=None,
                    )
                ],
            )

            result = service.generate_chunks(root)

            chunk = result.chunks[0].to_dict()
            self.assertEqual(
                set(chunk),
                {
                    "chunk_id",
                    "source_doc_id",
                    "source_doc_name",
                    "page_start",
                    "page_end",
                    "heading_path",
                    "block_ids",
                    "text",
                },
            )
            self.assertEqual(chunk["source_doc_id"], "doc_1")
            self.assertEqual(chunk["source_doc_name"], "paper.md")
            self.assertIsNone(chunk["page_start"])
            self.assertIsNone(chunk["page_end"])
            self.assertEqual(chunk["heading_path"], ["Intro"])
            self.assertEqual(chunk["text"], "Contract text.")

    def test_chunk_traceability_to_document_and_source_blocks(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            service = KnowledgeBaseService()
            service.init(root)
            _write_blocks(
                root,
                [
                    DocumentBlock.create(
                        source_doc_id="doc_trace",
                        source_doc_name="trace.pdf",
                        page=2,
                        order=0,
                        block_type="paragraph",
                        heading_path=(),
                        text="First source block.",
                        bbox=None,
                    ),
                    DocumentBlock.create(
                        source_doc_id="doc_trace",
                        source_doc_name="trace.pdf",
                        page=3,
                        order=1,
                        block_type="paragraph",
                        heading_path=(),
                        text="Second source block.",
                        bbox=None,
                    ),
                ],
            )

            chunk = service.generate_chunks(root).chunks[0]

            self.assertEqual(chunk.source_doc_id, "doc_trace")
            self.assertEqual(chunk.source_doc_name, "trace.pdf")
            self.assertEqual(chunk.page_start, 2)
            self.assertEqual(chunk.page_end, 3)
            self.assertEqual(len(chunk.block_ids), 2)

    def test_empty_blocks_have_clear_empty_chunks_behavior(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            service = KnowledgeBaseService()
            service.init(root)

            result = service.generate_chunks(root)

            self.assertEqual(result.chunks, ())
            self.assertEqual((root / ".kcontext" / "chunks.jsonl").read_text(encoding="utf-8"), "")

    def test_blocks_without_text_generate_empty_chunks_file(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            service = KnowledgeBaseService()
            service.init(root)
            store = LocalKnowledgeBaseStore()
            paths = store.paths(root)
            store.append_record(
                paths.blocks_path,
                {
                    "block_id": "block_empty",
                    "source_doc_id": "doc_empty",
                    "source_doc_name": "empty.txt",
                    "page": None,
                    "order": 0,
                    "block_type": "paragraph",
                    "heading_path": [],
                    "text": "   ",
                    "bbox": None,
                },
            )

            result = service.generate_chunks(root)

            self.assertEqual(result.chunks, ())
            self.assertEqual((root / ".kcontext" / "chunks.jsonl").read_text(encoding="utf-8"), "")


def _write_blocks(root: Path, blocks: list[DocumentBlock]) -> None:
    store = LocalKnowledgeBaseStore()
    paths = store.paths(root)
    store.replace_records(paths.blocks_path, (block.to_dict() for block in blocks))


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


if __name__ == "__main__":
    unittest.main()
