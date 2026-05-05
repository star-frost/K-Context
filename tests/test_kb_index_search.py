from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from k_context.application.kb_service import KnowledgeBaseService


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"


class KbIndexSearchTests(unittest.TestCase):
    def test_kb_index_generates_chunks_jsonl(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "notes.txt"
            source.write_text("Alpha search text.\n\nBeta text.", encoding="utf-8")
            service = KnowledgeBaseService()
            service.init(root)
            service.add(root, source)

            completed = _run_cli("index", "--root", str(root))

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("Index generated:", completed.stdout)
            self.assertIn("chunk_count: 1", completed.stdout)
            self.assertIn("Alpha search text", (root / ".kcontext" / "chunks.jsonl").read_text(encoding="utf-8"))

    def test_kb_search_returns_relevant_chunk_from_chunks_jsonl(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "notes.md"
            source.write_text("# Research\n\nAlpha keyword appears here.\n\nOther material.", encoding="utf-8")
            service = KnowledgeBaseService()
            service.init(root)
            add_result = service.add(root, source)
            service.generate_chunks(root)

            completed = _run_cli("search", "Alpha", "--root", str(root))

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("Search results: 1", completed.stdout)
            self.assertIn(add_result.document.doc_id, completed.stdout)
            self.assertIn("source_doc_name: notes.md", completed.stdout)
            self.assertIn("score:", completed.stdout)
            self.assertIn("block_ids:", completed.stdout)
            self.assertIn("Alpha keyword", completed.stdout)

    def test_search_has_clear_message_when_chunks_are_empty(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            KnowledgeBaseService().init(root)

            completed = _run_cli("search", "anything", "--root", str(root))

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("No chunks available", completed.stdout)

    def test_search_has_clear_message_when_no_match(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "notes.txt"
            source.write_text("Only banana content.", encoding="utf-8")
            service = KnowledgeBaseService()
            service.init(root)
            service.add(root, source)
            service.generate_chunks(root)

            completed = _run_cli("search", "zebra", "--root", str(root))

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("No matching chunks found.", completed.stdout)

    def test_top_k_limits_results(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            service = KnowledgeBaseService()
            service.init(root)
            for index in range(3):
                source = root / f"note_{index}.txt"
                source.write_text(f"shared term document {index}.", encoding="utf-8")
                service.add(root, source)
            service.generate_chunks(root)

            completed = _run_cli("search", "shared", "--root", str(root), "--top-k", "2")

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("Search results: 2", completed.stdout)
            self.assertIn("[1]", completed.stdout)
            self.assertIn("[2]", completed.stdout)
            self.assertNotIn("[3]", completed.stdout)

    def test_cli_scope_exposes_search_and_ask(self) -> None:
        completed = _run_cli("--help")

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("index", completed.stdout)
        self.assertIn("search", completed.stdout)
        self.assertIn("ask", completed.stdout)


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


if __name__ == "__main__":
    unittest.main()
