"""有依据 LLM 回答生成的 prompt 组装。"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from k_context.application.llm_client import LLMMessage, LLMRequest


DEFAULT_SYSTEM_INSTRUCTION = (
    "你是 K-Context 本地知识库助手。只能基于本次提供的 sources 回答；"
    "不得编造知识库外内容；证据不足时必须明确说明无法回答；"
    "回答应保留来源意识，并在适合时引用 source_doc_name 或 chunk_id。"
)
DEFAULT_MAX_QUOTE_CHARS = 500
DEFAULT_MAX_CONTEXT_CHARS = 6000

SENSITIVE_PLACEHOLDER = "[REDACTED]"
SENSITIVE_PATTERNS = (
    re.compile(r"KCONTEXT_LLM_API_KEY\s*=\s*\S+", re.IGNORECASE),
    re.compile(r"KCONTEXT_LLM_BASE_URL\s*=\s*\S+", re.IGNORECASE),
    re.compile(r"KCONTEXT_LLM_MODEL\s*=\s*\S+", re.IGNORECASE),
    re.compile(r"\bKCONTEXT_LLM_API_KEY\b", re.IGNORECASE),
    re.compile(r"\bKCONTEXT_LLM_BASE_URL\b", re.IGNORECASE),
    re.compile(r"\bKCONTEXT_LLM_MODEL\b", re.IGNORECASE),
    re.compile(r"Authorization\s*:\s*Bearer\s+\S+", re.IGNORECASE),
    re.compile(r"Bearer\s+[A-Za-z0-9._~+/=-]{8,}", re.IGNORECASE),
    re.compile(r"(api[_-]?key|apikey|secret)\s*[:=]\s*\S+", re.IGNORECASE),
)


class PromptBuilderError(RuntimeError):
    """当无法安全构建有依据 prompt 时抛出。"""


class RetrievedSource(Protocol):
    """prompt 组装所需的检索命中字段。"""

    chunk_id: str
    source_doc_id: str
    source_doc_name: str
    score: float
    retrieval_mode: str
    block_ids: tuple[str, ...]
    text: str
    metadata: dict[str, object]


@dataclass(frozen=True)
class PromptBuilder:
    """根据问题和已检索来源构建 LLMRequest 对象。"""

    default_system_instruction: str = DEFAULT_SYSTEM_INSTRUCTION
    max_quote_chars: int = DEFAULT_MAX_QUOTE_CHARS
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS

    def __post_init__(self) -> None:
        if self.max_quote_chars <= 0:
            raise ValueError("max_quote_chars must be positive.")
        if self.max_context_chars <= 0:
            raise ValueError("max_context_chars must be positive.")

    def build(
        self,
        *,
        question: str,
        sources: Sequence[RetrievedSource | Mapping[str, Any]],
        system_instruction: str | None = None,
    ) -> LLMRequest:
        """根据检索搜索结果或来源映射创建有依据 LLMRequest。"""

        sanitized_question = _redact_sensitive_text(question).strip()
        if not sanitized_question:
            raise PromptBuilderError("Cannot build LLMRequest from an empty question.")
        if not sources:
            raise PromptBuilderError("Cannot build LLMRequest without retrieved sources.")

        source_payloads = tuple(
            self._source_payload(source, index=index)
            for index, source in enumerate(sources, start=1)
        )
        system_content = _redact_sensitive_text(
            system_instruction or self.default_system_instruction
        ).strip()
        if not system_content:
            raise PromptBuilderError("Cannot build LLMRequest with an empty system instruction.")

        user_content = self._user_message(
            question=sanitized_question,
            sources=source_payloads,
        )
        return LLMRequest(
            question=sanitized_question,
            messages=(
                LLMMessage(role="system", content=system_content),
                LLMMessage(role="user", content=user_content),
            ),
            sources=source_payloads,
            metadata={"source_count": len(source_payloads)},
        )

    def _source_payload(
        self,
        source: RetrievedSource | Mapping[str, Any],
        *,
        index: int,
    ) -> Mapping[str, Any]:
        metadata = _metadata(source)
        chunk = getattr(source, "chunk", None)

        chunk_id = _required_text(_value(source, "chunk_id"), "chunk_id")
        source_doc_id = _required_text(_value(source, "source_doc_id"), "source_doc_id")
        source_doc_name = _required_text(
            _value(source, "source_doc_name"),
            "source_doc_name",
        )
        raw_block_ids = _value(source, "block_ids")
        block_ids = tuple(str(item) for item in (raw_block_ids or ()))

        text = _first_text(
            _value(source, "text"),
            _value(source, "quote"),
        )
        quote = _truncate(
            _redact_sensitive_text(text).strip(),
            self.max_quote_chars,
        ) if text is not None and text.strip() else None

        return {
            "source_index": index,
            "chunk_id": chunk_id,
            "source_doc_id": source_doc_id,
            "source_doc_name": source_doc_name,
            "score": _optional_float(_value(source, "score")),
            "block_ids": list(block_ids),
            "page_start": _page_value(source, chunk, metadata, "page_start"),
            "page_end": _page_value(source, chunk, metadata, "page_end"),
            "quote": quote,
            "retrieval_mode": _optional_text(_value(source, "retrieval_mode")),
        }

    def _user_message(
        self,
        *,
        question: str,
        sources: tuple[Mapping[str, Any], ...],
    ) -> str:
        context_sources = self._fit_context_sources(
            tuple(_prompt_context_source(source) for source in sources)
        )
        context = json.dumps(context_sources, ensure_ascii=False, indent=2)
        return "\n".join(
            (
                "请基于以下 sources 回答问题。",
                "要求：只能使用 sources 中的信息；不得使用外部知识；"
                "如果 sources 不能支持答案，请回答“证据不足，无法回答”。",
                "",
                f"问题：{question}",
                "",
                "sources:",
                context,
            )
        )

    def _fit_context_sources(
        self,
        sources: tuple[Mapping[str, Any], ...],
    ) -> tuple[Mapping[str, Any], ...]:
        """缩短引用文本，同时保留最小引用标识。"""

        if len(json.dumps(sources, ensure_ascii=False)) <= self.max_context_chars:
            return sources

        quote_budget = max(40, self.max_context_chars // max(1, len(sources)) // 2)
        shortened = tuple(
            _copy_with_shorter_quote(source, max_quote_chars=quote_budget)
            for source in sources
        )
        if len(json.dumps(shortened, ensure_ascii=False)) <= self.max_context_chars:
            return shortened

        return tuple(_copy_with_shorter_quote(source, max_quote_chars=0) for source in sources)


def _metadata(source: RetrievedSource | Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = _value(source, "metadata")
    return metadata if isinstance(metadata, Mapping) else {}


def _value(source: RetrievedSource | Mapping[str, Any], field_name: str) -> Any:
    if isinstance(source, Mapping):
        return source.get(field_name)
    return getattr(source, field_name, None)


def _required_text(value: Any, field_name: str) -> str:
    text = _optional_text(value)
    if text is None:
        raise PromptBuilderError(f"Cannot build source payload without {field_name}.")
    return text


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return _redact_sensitive_text(text) if text else None


def _first_text(*values: Any) -> str | None:
    for value in values:
        text = _optional_text(value)
        if text is not None:
            return text
    return None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise PromptBuilderError("Source score must be numeric when provided.")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise PromptBuilderError("Source score must be numeric when provided.") from exc


def _page_value(
    source: RetrievedSource | Mapping[str, Any],
    chunk: object | None,
    metadata: Mapping[str, Any],
    field_name: str,
) -> int | None:
    for value in (
        _value(source, field_name),
        getattr(chunk, field_name, None) if chunk is not None else None,
        metadata.get(field_name),
    ):
        if value is None:
            continue
        if isinstance(value, bool):
            raise PromptBuilderError(f"Source {field_name} must be an integer or null.")
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise PromptBuilderError(
                f"Source {field_name} must be an integer or null."
            ) from exc
    return None


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def _copy_with_shorter_quote(
    source: Mapping[str, Any],
    *,
    max_quote_chars: int,
) -> Mapping[str, Any]:
    copied = dict(source)
    quote = copied.get("quote")
    if isinstance(quote, str):
        copied["quote"] = _truncate(quote, max_quote_chars) if max_quote_chars > 0 else None
    return copied


def _prompt_context_source(source: Mapping[str, Any]) -> Mapping[str, Any]:
    """仅返回有助于回答生成和最小引用的字段。

    完整来源载荷仍保留在 ``LLMRequest.sources`` 中，供应用侧追溯；
    但 LLM prompt 不应把上下文花在分数、block-id 列表、检索模式
    或其他运行元数据上。
    """

    return {
        "source_index": source.get("source_index"),
        "chunk_id": source.get("chunk_id"),
        "source_doc_name": source.get("source_doc_name"),
        "page_start": source.get("page_start"),
        "page_end": source.get("page_end"),
        "quote": source.get("quote"),
    }


def _redact_sensitive_text(text: str) -> str:
    redacted = text
    for pattern in SENSITIVE_PATTERNS:
        redacted = pattern.sub(SENSITIVE_PLACEHOLDER, redacted)
    return redacted
