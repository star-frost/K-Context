"""本地知识库的追加式指标事件记录。"""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator, Mapping
from uuid import uuid4

from k_context.application.mcp_contracts import MCP_METRICS_EVENT_TYPES
from k_context.infrastructure.storage.local_store import LocalKnowledgeBaseStore


SUPPORTED_METRICS_EVENT_TYPES = {
    "index",
    "cleaning",
    "chunking",
    "embedding",
    "vector_upsert",
    "retrieval",
    "ask",
    "llm_call",
    "token_usage",
    "ask_fallback",
    "evaluation",
    "recall_at_k",
    *MCP_METRICS_EVENT_TYPES,
}
SUCCESS_STATUS = "success"
FAILURE_STATUS = "failure"


@dataclass(frozen=True)
class MetricsEvent:
    """符合本地 metrics.jsonl 契约的可序列化指标事件。"""

    event_id: str
    event_type: str
    operation: str
    started_at: str
    ended_at: str
    duration_ms: float
    status: str
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str | None = None
    retrieval_mode: str | None = None
    top_k: int | None = None
    embedding_model: str | None = None
    vector_store_type: str | None = None
    token_usage: dict[str, Any] | None = None
    related_session_id: str | None = None
    related_query_id: str | None = None

    @classmethod
    def create(
        cls,
        *,
        event_type: str,
        operation: str,
        started_at: str,
        ended_at: str,
        duration_ms: float,
        status: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> "MetricsEvent":
        checked_metadata = dict(metadata or {})
        return cls(
            event_id=f"metrics_{uuid4().hex}",
            event_type=_require_supported_event_type(event_type),
            operation=_require_non_empty("operation", operation),
            started_at=started_at,
            ended_at=ended_at,
            duration_ms=max(0.0, round(float(duration_ms), 3)),
            status=_require_supported_status(status),
            metadata=checked_metadata,
            timestamp=started_at,
            retrieval_mode=_optional_str(checked_metadata.get("retrieval_mode")),
            top_k=_optional_int(checked_metadata.get("top_k")),
            embedding_model=_optional_str(checked_metadata.get("embedding_model")),
            vector_store_type=_optional_str(checked_metadata.get("vector_store_type")),
            token_usage=_optional_dict(checked_metadata.get("token_usage")),
            related_session_id=_optional_str(checked_metadata.get("related_session_id")),
            related_query_id=_optional_str(checked_metadata.get("related_query_id")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "operation": self.operation,
            "timestamp": self.timestamp,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "retrieval_mode": self.retrieval_mode,
            "top_k": self.top_k,
            "embedding_model": self.embedding_model,
            "vector_store_type": self.vector_store_type,
            "token_usage": self.token_usage,
            "related_session_id": self.related_session_id,
            "related_query_id": self.related_query_id,
            "metadata": dict(self.metadata),
        }


class MetricsCollector:
    """将指标事件写入 metrics.jsonl，不覆盖既有记录。"""

    def __init__(self, metrics_path: Path) -> None:
        self.metrics_path = metrics_path.expanduser().resolve()

    @classmethod
    def from_root(
        cls,
        root: Path,
        *,
        store: LocalKnowledgeBaseStore | None = None,
    ) -> "MetricsCollector":
        paths = (store or LocalKnowledgeBaseStore()).require_initialized(root)
        return cls(paths.metrics_path)

    def record_success(
        self,
        *,
        event_type: str,
        operation: str,
        started_at: str | None = None,
        ended_at: str | None = None,
        duration_ms: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> MetricsEvent:
        return self.record_event(
            event_type=event_type,
            operation=operation,
            status=SUCCESS_STATUS,
            started_at=started_at,
            ended_at=ended_at,
            duration_ms=duration_ms,
            metadata=metadata,
        )

    def record_failure(
        self,
        *,
        event_type: str,
        operation: str,
        error_message: str,
        started_at: str | None = None,
        ended_at: str | None = None,
        duration_ms: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> MetricsEvent:
        merged_metadata = dict(metadata or {})
        merged_metadata["error_message"] = error_message
        return self.record_event(
            event_type=event_type,
            operation=operation,
            status=FAILURE_STATUS,
            started_at=started_at,
            ended_at=ended_at,
            duration_ms=duration_ms,
            metadata=merged_metadata,
        )

    def record_event(
        self,
        *,
        event_type: str,
        operation: str,
        status: str,
        started_at: str | None = None,
        ended_at: str | None = None,
        duration_ms: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> MetricsEvent:
        resolved_started_at = started_at or _utc_now()
        resolved_ended_at = ended_at or _utc_now()
        event = MetricsEvent.create(
            event_type=event_type,
            operation=operation,
            started_at=resolved_started_at,
            ended_at=resolved_ended_at,
            duration_ms=duration_ms if duration_ms is not None else 0.0,
            status=status,
            metadata=metadata,
        )
        self.append(event)
        return event

    @contextmanager
    def measure(
        self,
        *,
        event_type: str,
        operation: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> Iterator[None]:
        started_at = _utc_now()
        start = perf_counter()
        try:
            yield
        except Exception as exc:
            ended_at = _utc_now()
            self.record_failure(
                event_type=event_type,
                operation=operation,
                error_message=str(exc),
                started_at=started_at,
                ended_at=ended_at,
                duration_ms=_elapsed_ms(start),
                metadata=metadata,
            )
            raise
        else:
            ended_at = _utc_now()
            self.record_success(
                event_type=event_type,
                operation=operation,
                started_at=started_at,
                ended_at=ended_at,
                duration_ms=_elapsed_ms(start),
                metadata=metadata,
            )

    def append(self, event: MetricsEvent) -> None:
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with self.metrics_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")

    def read_events(self) -> tuple[dict[str, Any], ...]:
        if not self.metrics_path.exists():
            return ()
        events = []
        for line in self.metrics_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                events.append(json.loads(line))
        return tuple(events)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _elapsed_ms(start: float) -> float:
    return max(0.0, round((perf_counter() - start) * 1000, 3))


def _require_supported_event_type(event_type: str) -> str:
    value = _require_non_empty("event_type", event_type)
    if value not in SUPPORTED_METRICS_EVENT_TYPES:
        raise ValueError(
            "Unsupported metrics event_type: "
            f"{value}. Supported values: {', '.join(sorted(SUPPORTED_METRICS_EVENT_TYPES))}."
        )
    return value


def _require_supported_status(status: str) -> str:
    value = _require_non_empty("status", status)
    if value not in {SUCCESS_STATUS, FAILURE_STATUS}:
        raise ValueError("Metrics status must be success or failure.")
    return value


def _require_non_empty(field_name: str, value: str) -> str:
    checked = str(value).strip()
    if not checked:
        raise ValueError(f"{field_name} must not be empty.")
    return checked


def _optional_str(value: Any) -> str | None:
    return str(value) if value is not None else None


def _optional_int(value: Any) -> int | None:
    return int(value) if value is not None else None


def _optional_dict(value: Any) -> dict[str, Any] | None:
    return dict(value) if isinstance(value, Mapping) else None
