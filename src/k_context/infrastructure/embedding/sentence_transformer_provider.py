"""基于 sentence-transformers 的 embedding provider。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from k_context.application.config_service import ConfigService
from k_context.application.embedding_provider import (
    DEFAULT_BGE_MODEL,
    SUPPORTED_EMBEDDING_DEVICES,
    EmbeddingInput,
    EmbeddingProviderError,
)
from k_context.domain.models import EmbeddingRecord, QueryEmbedding


BGE_MODEL_ALIASES = {
    DEFAULT_BGE_MODEL: "BAAI/bge-m3",
    "bge-base-en-v1.5": "BAAI/bge-base-en-v1.5",
}


class SentenceTransformerEmbeddingProvider:
    """由 sentence-transformers 支撑的 embedding provider。"""

    def __init__(
        self,
        *,
        embedding_model: str = DEFAULT_BGE_MODEL,
        embedding_device: str = "auto",
        model_loader: Callable[..., Any] | None = None,
    ) -> None:
        if embedding_device not in SUPPORTED_EMBEDDING_DEVICES:
            raise ValueError(
                f"Unsupported embedding_device: {embedding_device}. "
                "Supported devices: auto, cpu, cuda"
            )
        self.embedding_model = embedding_model
        self.embedding_device = embedding_device
        self.model_name_or_path = self._resolve_model_name_or_path(embedding_model)
        self._model = self._load_model(model_loader)
        self.embedding_dim = self._detect_embedding_dim()

    @classmethod
    def from_config(
        cls,
        root: Path,
        *,
        config_service: ConfigService | None = None,
        model_loader: Callable[..., Any] | None = None,
    ) -> "SentenceTransformerEmbeddingProvider":
        config = (config_service or ConfigService()).load(root)
        return cls(
            embedding_model=config.embedding_model,
            embedding_device=config.embedding_device,
            model_loader=model_loader,
        )

    def embed_documents(self, inputs: tuple[EmbeddingInput, ...]) -> tuple[EmbeddingRecord, ...]:
        if not inputs:
            return ()
        checked_inputs = tuple(
            EmbeddingInput(source_id=item.source_id, text=self._require_text(item.text))
            for item in inputs
        )
        vectors = self._encode_documents(tuple(item.text for item in checked_inputs))
        return tuple(
            EmbeddingRecord.create(
                chunk_id=item.source_id,
                text=item.text,
                embedding=vector,
                embedding_model=self.embedding_model,
            )
            for item, vector in zip(checked_inputs, vectors, strict=True)
        )

    def embed_query(self, text: str) -> QueryEmbedding:
        checked_text = self._require_text(text)
        return QueryEmbedding.create(
            text=checked_text,
            embedding=self._encode_query(checked_text),
            embedding_model=self.embedding_model,
        )

    def _resolve_model_name_or_path(self, embedding_model: str) -> str:
        return BGE_MODEL_ALIASES.get(embedding_model, embedding_model)

    def _load_model(self, model_loader: Callable[..., Any] | None) -> Any:
        loader = model_loader or self._default_model_loader()
        try:
            if self.embedding_device == "auto":
                return loader(self.model_name_or_path)
            return loader(self.model_name_or_path, device=self.embedding_device)
        except Exception as exc:  # pragma: no cover - exact backend exceptions vary.
            raise EmbeddingProviderError(
                "Failed to load sentence-transformers model "
                f"{self.model_name_or_path!r} for configured embedding_model "
                f"{self.embedding_model!r} on device {self.embedding_device!r}: {exc}"
            ) from exc

    def _default_model_loader(self) -> Callable[..., Any]:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingProviderError(
                "sentence-transformers is not installed. Install project dependencies before "
                "using SentenceTransformerEmbeddingProvider."
            ) from exc
        return SentenceTransformer

    def _detect_embedding_dim(self) -> int:
        if hasattr(self._model, "get_embedding_dimension"):
            dim = self._model.get_embedding_dimension()
            if dim:
                return int(dim)
        if hasattr(self._model, "get_sentence_embedding_dimension"):
            dim = self._model.get_sentence_embedding_dimension()
            if dim:
                return int(dim)
        vector = self._encode_query("dimension probe")
        return len(vector)

    def _encode_documents(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
        try:
            raw_vectors = self._model.encode(list(texts))
        except Exception as exc:
            raise EmbeddingProviderError(f"Failed to encode document embeddings: {exc}") from exc
        vectors = self._as_document_vectors(raw_vectors, expected_count=len(texts))
        self._validate_dimensions((vector for vector in vectors))
        return vectors

    def _encode_query(self, text: str) -> tuple[float, ...]:
        try:
            raw_vector = self._model.encode(text)
        except Exception as exc:
            raise EmbeddingProviderError(f"Failed to encode query embedding: {exc}") from exc
        vector = self._as_query_vector(raw_vector)
        if not vector:
            raise EmbeddingProviderError("Embedding model returned an empty query vector.")
        self._validate_dimensions((vector,))
        return vector

    def _as_document_vectors(self, raw_vectors: Any, *, expected_count: int) -> tuple[tuple[float, ...], ...]:
        value = self._to_builtin(raw_vectors)
        if expected_count == 1 and self._is_number_sequence(value):
            value = [value]
        if not isinstance(value, list) or len(value) != expected_count:
            raise EmbeddingProviderError(
                f"Embedding model returned {len(value) if isinstance(value, list) else 'non-list'} "
                f"document vectors for {expected_count} inputs."
            )
        return tuple(self._coerce_vector(item) for item in value)

    def _as_query_vector(self, raw_vector: Any) -> tuple[float, ...]:
        value = self._to_builtin(raw_vector)
        if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
            value = value[0]
        return self._coerce_vector(value)

    def _coerce_vector(self, value: Any) -> tuple[float, ...]:
        value = self._to_builtin(value)
        if not self._is_number_sequence(value):
            raise EmbeddingProviderError("Embedding model returned a non-numeric vector.")
        return tuple(float(item) for item in value)

    def _validate_dimensions(self, vectors: Any) -> None:
        for vector in vectors:
            if len(vector) != self.embedding_dim:
                raise EmbeddingProviderError(
                    f"Embedding dimension mismatch: expected {self.embedding_dim}, got {len(vector)}."
                )

    def _to_builtin(self, value: Any) -> Any:
        return value.tolist() if hasattr(value, "tolist") else value

    def _is_number_sequence(self, value: Any) -> bool:
        return isinstance(value, list) and all(isinstance(item, int | float) for item in value)

    def _require_text(self, text: str) -> str:
        if not text.strip():
            raise ValueError("Cannot embed empty text.")
        return text
