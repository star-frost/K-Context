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
from k_context.presentation.cli import build_parser


class KbInitTests(unittest.TestCase):
    def test_service_creates_local_kb_layout(self) -> None:
        with TemporaryDirectory() as tmp:
            result = KnowledgeBaseService().init(Path(tmp))
            kb_root = Path(tmp) / ".kcontext"

            self.assertEqual(result.kb_root, kb_root.resolve())
            self.assertFalse(result.already_initialized)
            self.assertTrue((kb_root / "config.json").is_file())
            self.assertTrue((kb_root / "metadata.jsonl").is_file())
            self.assertTrue((kb_root / "chunks.jsonl").is_file())
            self.assertTrue((kb_root / "sessions.jsonl").is_file())
            self.assertTrue((kb_root / "index").is_dir())
            self.assertEqual(
                {path.name for path in kb_root.iterdir()},
                {"config.json", "metadata.jsonl", "chunks.jsonl", "sessions.jsonl", "index"},
            )

            config = json.loads((kb_root / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["schema_version"], 1)
            self.assertEqual(config["storage"]["metadata"], "metadata.jsonl")
            self.assertEqual(config["storage"]["chunks"], "chunks.jsonl")

    def test_service_init_is_idempotent(self) -> None:
        with TemporaryDirectory() as tmp:
            service = KnowledgeBaseService()
            first = service.init(Path(tmp))
            second = service.init(Path(tmp))

            self.assertFalse(first.already_initialized)
            self.assertTrue(second.already_initialized)
            self.assertEqual(second.created_paths, ())

    def test_cli_init_creates_layout(self) -> None:
        with TemporaryDirectory() as tmp:
            env = os.environ.copy()
            env["PYTHONPATH"] = str(SRC_ROOT)
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "k_context.presentation.cli",
                    "init",
                    "--root",
                    tmp,
                ],
                check=False,
                capture_output=True,
                text=True,
                env=env,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("Knowledge base initialized", completed.stdout)
            self.assertTrue((Path(tmp) / ".kcontext" / "config.json").is_file())

    def test_cli_scope_exposes_only_implemented_commands(self) -> None:
        help_text = build_parser().format_help()

        self.assertIn("init", help_text)
        self.assertIn("add", help_text)
        self.assertIn("index", help_text)
        self.assertIn("search", help_text)
        self.assertIn("ask", help_text)


if __name__ == "__main__":
    unittest.main()
