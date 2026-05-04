"""Local on-disk storage layout for a single-user knowledge base."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


KB_DIR_NAME = ".kcontext"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class KbInitRecord:
    """Storage-layer record describing the result of initialization."""

    kb_root: Path
    created_paths: tuple[Path, ...]
    already_initialized: bool


@dataclass(frozen=True)
class KnowledgeBasePaths:
    """Resolved paths for the local knowledge-base persistence files."""

    kb_root: Path
    metadata_path: Path
    chunks_path: Path
    sessions_path: Path
    index_dir: Path
    blocks_path: Path


class LocalKnowledgeBaseStore:
    """Creates the minimum local persistence structure required by the docs."""

    def initialize(self, root: Path) -> KbInitRecord:
        project_root = root.expanduser().resolve()
        kb_root = project_root / KB_DIR_NAME
        config_path = kb_root / "config.json"

        expected_dirs = (kb_root, kb_root / "index")
        expected_files = (
            config_path,
            kb_root / "metadata.jsonl",
            kb_root / "chunks.jsonl",
            kb_root / "sessions.jsonl",
        )

        was_initialized = config_path.exists()
        created: list[Path] = []

        project_root.mkdir(parents=True, exist_ok=True)
        for directory in expected_dirs:
            if not directory.exists():
                directory.mkdir(parents=True)
                created.append(directory)

        if not config_path.exists():
            self._write_config(config_path)
            created.append(config_path)

        for file_path in expected_files:
            if file_path == config_path:
                continue
            if not file_path.exists():
                file_path.write_text("", encoding="utf-8")
                created.append(file_path)

        return KbInitRecord(
            kb_root=kb_root,
            created_paths=tuple(created),
            already_initialized=was_initialized and not created,
        )

    def require_initialized(self, root: Path) -> KnowledgeBasePaths:
        paths = self.paths(root)
        if not (paths.kb_root / "config.json").is_file():
            raise FileNotFoundError(
                f"Knowledge base is not initialized under {root}. Run `kb init --root {root}` first."
            )
        return paths

    def paths(self, root: Path) -> KnowledgeBasePaths:
        kb_root = root.expanduser().resolve() / KB_DIR_NAME
        return KnowledgeBasePaths(
            kb_root=kb_root,
            metadata_path=kb_root / "metadata.jsonl",
            chunks_path=kb_root / "chunks.jsonl",
            sessions_path=kb_root / "sessions.jsonl",
            index_dir=kb_root / "index",
            blocks_path=kb_root / "blocks.jsonl",
        )

    def append_record(self, path: Path, record: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def append_records(self, path: Path, records: Iterable[Mapping[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def read_records(self, path: Path) -> tuple[dict[str, Any], ...]:
        if not path.is_file():
            return ()
        return tuple(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )

    def replace_records(self, path: Path, records: Iterable[Mapping[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _write_config(self, path: Path) -> None:
        now = datetime.now(timezone.utc).isoformat()
        payload = {
            "schema_version": SCHEMA_VERSION,
            "created_at": now,
            "storage": {
                "metadata": "metadata.jsonl",
                "chunks": "chunks.jsonl",
                "sessions": "sessions.jsonl",
                "index_dir": "index",
            },
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
