from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from k_context.application.answer_service import EVIDENCE_LEVELS
from k_context.application.kb_service import KnowledgeBaseService


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"


class KbAskTests(unittest.TestCase):
    def test_ask_outputs_answer_and_sources_for_relevant_chunk(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "notes.md"
            source.write_text("# Notes\n\n样例关键词 means grounded evidence.", encoding="utf-8")
            service = KnowledgeBaseService()
            service.init(root)
            add_result = service.add(root, source)
            service.generate_chunks(root)

            completed = _run_cli("ask", "样例关键词 是什么", "--root", str(root))

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("answer:", completed.stdout)
            self.assertIn("evidence_level:", completed.stdout)
            self.assertIn("sources:", completed.stdout)
            self.assertIn(add_result.document.doc_id, completed.stdout)
            self.assertIn("source_doc_name: notes.md", completed.stdout)
            self.assertIn("chunk_id:", completed.stdout)
            self.assertIn("block_ids:", completed.stdout)

    def test_ask_reports_insufficient_evidence_when_no_chunks(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            KnowledgeBaseService().init(root)

            completed = _run_cli("ask", "anything", "--root", str(root))

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("evidence_level: 证据不足", completed.stdout)
            self.assertIn("当前知识库依据不足", completed.stdout)
            self.assertIn("sources:", completed.stdout)
            self.assertIn("[]", completed.stdout)

    def test_ask_reports_insufficient_evidence_when_no_match(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "notes.txt"
            source.write_text("banana only.", encoding="utf-8")
            service = KnowledgeBaseService()
            service.init(root)
            service.add(root, source)
            service.generate_chunks(root)

            completed = _run_cli("ask", "zebra", "--root", str(root))

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("evidence_level: 证据不足", completed.stdout)
            self.assertIn("当前知识库依据不足", completed.stdout)

    def test_ask_sources_trace_to_chunk_and_document(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "trace.txt"
            source.write_text("traceable source content.", encoding="utf-8")
            service = KnowledgeBaseService()
            service.init(root)
            add_result = service.add(root, source)
            chunk_result = service.generate_chunks(root)

            answer = service.ask(root, "traceable")

            self.assertEqual(answer.sources[0].source_doc_id, add_result.document.doc_id)
            self.assertEqual(answer.sources[0].chunk_id, chunk_result.chunks[0].chunk_id)
            self.assertEqual(answer.sources[0].block_ids, chunk_result.chunks[0].block_ids)

    def test_evidence_level_is_limited_to_documented_values(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "notes.txt"
            source.write_text("alpha evidence.", encoding="utf-8")
            service = KnowledgeBaseService()
            service.init(root)
            service.add(root, source)
            service.generate_chunks(root)

            answer = service.ask(root, "alpha")

            self.assertIn(answer.evidence_level, EVIDENCE_LEVELS)
            self.assertEqual(set(EVIDENCE_LEVELS), {"证据不足", "基本充分", "充分"})

    def test_cli_scope_includes_ask_without_external_model_flags(self) -> None:
        completed = _run_cli("--help")

        self.assertEqual(completed.returncode, 0, completed.stderr)
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
