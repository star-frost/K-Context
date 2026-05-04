from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from k_context.application.kb_service import KnowledgeBaseService


class KbAddTests(unittest.TestCase):
    def test_add_txt_writes_document_metadata_and_blocks(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "note.txt"
            source.write_text("Alpha knowledge.\n\nBeta knowledge.", encoding="utf-8")

            KnowledgeBaseService().init(root)
            result = KnowledgeBaseService().add(root, source)

            metadata_records = _read_jsonl(root / ".kcontext" / "metadata.jsonl")
            block_records = _read_jsonl(root / ".kcontext" / "blocks.jsonl")

            self.assertEqual(result.document.file_name, "note.txt")
            self.assertEqual(result.document.file_type, "txt")
            self.assertEqual(result.document.status, "已解析")
            self.assertEqual(metadata_records, [result.document.to_dict()])
            self.assertEqual(len(block_records), 2)
            self.assertEqual(block_records[0]["source_doc_id"], result.document.doc_id)
            self.assertEqual(block_records[0]["source_doc_name"], "note.txt")
            self.assertEqual(block_records[0]["block_type"], "paragraph")
            self.assertEqual(block_records[0]["heading_path"], [])
            self.assertIsNone(block_records[0]["page"])
            self.assertIsNone(block_records[0]["bbox"])

    def test_add_markdown_preserves_heading_path(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "paper.md"
            source.write_text("# Intro\n\nImportant finding.\n\n## Detail\n\nMore evidence.", encoding="utf-8")

            KnowledgeBaseService().init(root)
            KnowledgeBaseService().add(root, source)

            blocks = _read_jsonl(root / ".kcontext" / "blocks.jsonl")
            self.assertEqual(blocks[0]["block_type"], "title")
            self.assertEqual(blocks[0]["heading_path"], ["Intro"])
            self.assertEqual(blocks[1]["heading_path"], ["Intro"])
            self.assertEqual(blocks[3]["heading_path"], ["Intro", "Detail"])

    def test_add_pdf_extracts_plain_stream_text(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "plain.pdf"
            _write_minimal_pdf(source, "Plain PDF knowledge")

            KnowledgeBaseService().init(root)
            result = KnowledgeBaseService().add(root, source)

            blocks = _read_jsonl(root / ".kcontext" / "blocks.jsonl")
            self.assertEqual(result.document.file_type, "pdf")
            self.assertEqual(blocks[0]["page"], 1)
            self.assertIn("Plain PDF knowledge", blocks[0]["text"])

    def test_add_missing_file_returns_cli_error(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            KnowledgeBaseService().init(root)

            completed = _run_cli("add", str(root / "missing.txt"), "--root", str(root))

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("Error:", completed.stdout)
            self.assertIn("File does not exist", completed.stdout)

    def test_add_unsupported_file_returns_cli_error(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "paper.docx"
            source.write_text("not supported", encoding="utf-8")
            KnowledgeBaseService().init(root)

            completed = _run_cli("add", str(source), "--root", str(root))

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("Unsupported file type", completed.stdout)

    def test_cli_add_outputs_document_summary(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "note.md"
            source.write_text("# Summary\n\nCLI add works.", encoding="utf-8")
            KnowledgeBaseService().init(root)

            completed = _run_cli("add", str(source), "--root", str(root))

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("Document added:", completed.stdout)
            self.assertIn("document_id:", completed.stdout)
            self.assertIn("file_name: note.md", completed.stdout)
            self.assertIn("file_type: md", completed.stdout)
            self.assertIn("metadata_path:", completed.stdout)

    def test_init_layout_is_unchanged_before_add(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            KnowledgeBaseService().init(root)

            self.assertEqual(
                {path.name for path in (root / ".kcontext").iterdir()},
                {"config.json", "metadata.jsonl", "chunks.jsonl", "sessions.jsonl", "index"},
            )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_ROOT)
    return subprocess.run(
        [sys.executable, "-m", "k_context.presentation.cli", *args],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def _write_minimal_pdf(path: Path, text: str) -> None:
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("latin-1")
    path.write_bytes(
        b"%PDF-1.4\n"
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
        b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n"
        b"3 0 obj << /Type /Page /Parent 2 0 R /Contents 4 0 R >> endobj\n"
        + f"4 0 obj << /Length {len(stream)} >> stream\n".encode("ascii")
        + stream
        + b"\nendstream endobj\n%%EOF\n"
    )


if __name__ == "__main__":
    unittest.main()
