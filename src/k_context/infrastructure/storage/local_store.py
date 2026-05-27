"""单用户知识库的本地磁盘存储布局。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from k_context.domain.models import DEFAULT_CONFIG_VALUES

KB_DIR_NAME = ".kcontext"
SCHEMA_VERSION = 2


@dataclass(frozen=True)
class KbInitRecord:
    """描述初始化结果的存储层记录。"""

    kb_root: Path
    created_paths: tuple[Path, ...]
    already_initialized: bool


@dataclass(frozen=True)
class KnowledgeBasePaths:
    """本地知识库持久化文件的解析后路径。"""

    kb_root: Path
    config_path: Path
    metadata_path: Path
    blocks_path: Path
    chunks_path: Path
    sessions_path: Path
    metrics_path: Path
    index_dir: Path
    chroma_dir: Path
    cleaned_blocks_path: Path


class LocalKnowledgeBaseStore:
    """创建文档要求的最小本地持久化结构。"""

    def initialize(self, root: Path) -> KbInitRecord:
        project_root = root.expanduser().resolve()
        kb_root = project_root / KB_DIR_NAME
        config_path = kb_root / "config.json"

        expected_dirs = (kb_root, kb_root / "index", kb_root / "index" / "chroma")
        expected_files = (
            config_path,
            kb_root / "metadata.jsonl",
            kb_root / "blocks.jsonl",
            kb_root / "cleaned_blocks.jsonl",
            kb_root / "chunks.jsonl",
            kb_root / "sessions.jsonl",
            kb_root / "metrics.jsonl",
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
            config_path=kb_root / "config.json",
            metadata_path=kb_root / "metadata.jsonl",
            blocks_path=kb_root / "blocks.jsonl",
            chunks_path=kb_root / "chunks.jsonl",
            sessions_path=kb_root / "sessions.jsonl",
            metrics_path=kb_root / "metrics.jsonl",
            index_dir=kb_root / "index",
            chroma_dir=kb_root / "index" / "chroma",
            cleaned_blocks_path=kb_root / "cleaned_blocks.jsonl",
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
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    records.append(json.loads(line))
        return tuple(records)

    def read_block_records(self, paths: KnowledgeBasePaths) -> tuple[dict[str, Any], ...]:
        return self.read_records(paths.blocks_path)

    def read_cleaned_block_records(self, paths: KnowledgeBasePaths) -> tuple[dict[str, Any], ...]:
        return self.read_records(paths.cleaned_blocks_path)

    def replace_cleaned_block_records(
        self,
        paths: KnowledgeBasePaths,
        records: Iterable[Mapping[str, Any]],
    ) -> None:
        self.replace_records(paths.cleaned_blocks_path, records)

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
            **DEFAULT_CONFIG_VALUES,
            "storage": {
                "metadata": "metadata.jsonl",
                "blocks": "blocks.jsonl",
                "cleaned_blocks": "cleaned_blocks.jsonl",
                "chunks": "chunks.jsonl",
                "sessions": "sessions.jsonl",
                "metrics": "metrics.jsonl",
                "index_dir": "index",
                "chroma_dir": "index/chroma",
            },
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
