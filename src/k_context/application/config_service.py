from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping

from k_context.domain.models import DEFAULT_CONFIG_VALUES, KContextConfig
from k_context.infrastructure.storage.local_store import KB_DIR_NAME


CONFIG_ENV_VARS = {
    "embedding_model": "KCONTEXT_EMBEDDING_MODEL",
    "embedding_device": "KCONTEXT_EMBEDDING_DEVICE",
    "vector_store_type": "KCONTEXT_VECTOR_STORE_TYPE",
    "chroma_persist_dir": "KCONTEXT_CHROMA_PERSIST_DIR",
    "chunking_strategy": "KCONTEXT_CHUNKING_STRATEGY",
    "cleaning_profile": "KCONTEXT_CLEANING_PROFILE",
    "retrieval_mode": "KCONTEXT_RETRIEVAL_MODE",
    "rag_method": "KCONTEXT_RAG_METHOD",
    "top_k": "KCONTEXT_TOP_K",
    "llm_base_url": "KCONTEXT_LLM_BASE_URL",
    "llm_model": "KCONTEXT_LLM_MODEL",
}


class ConfigService:
    """
    按默认值、环境变量和运行时覆盖层读取配置。

    """

    def load(
        self,
        root: Path,
        *,
        runtime_overrides: Mapping[str, object | None] | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> KContextConfig:
        raw_config = self._read_config(root)
        values = dict(DEFAULT_CONFIG_VALUES)
        values.update(self._known_config_values(raw_config))
        values.update(self._environment_overrides(environ or os.environ))
        if runtime_overrides:
            values.update(
                {
                    key: value
                    for key, value in runtime_overrides.items()
                    if key in DEFAULT_CONFIG_VALUES and value is not None
                }
            )
        return KContextConfig.from_mapping(values)

    def resolve_chroma_persist_dir(self, root: Path, config: KContextConfig) -> Path:
        persist_dir = Path(config.chroma_persist_dir)
        if persist_dir.is_absolute():
            return persist_dir
        return root.expanduser().resolve() / KB_DIR_NAME / persist_dir

    def _read_config(self, root: Path) -> dict[str, object]:
        config_path = root.expanduser().resolve() / KB_DIR_NAME / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(
                f"Knowledge base is not initialized under {root}. Run `kb init --root {root}` first."
            )
        return json.loads(config_path.read_text(encoding="utf-8"))

    def _known_config_values(self, raw_config: Mapping[str, object]) -> dict[str, object]:
        return {key: raw_config[key] for key in DEFAULT_CONFIG_VALUES if key in raw_config}

    def _environment_overrides(self, environ: Mapping[str, str]) -> dict[str, object]:
        return {
            key: environ[env_var]
            for key, env_var in CONFIG_ENV_VARS.items()
            if env_var in environ and environ[env_var] != ""
        }
