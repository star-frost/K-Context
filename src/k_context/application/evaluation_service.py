"""仅检索评估 Recall@k 的服务。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Iterable, Mapping, Protocol

from k_context.application.metrics_collector import MetricsCollector
from k_context.application.retrieval_service import DEFAULT_TOP_K, RetrievalResults, RetrievalService


CHUNK_RECALL_BASIS = "chunk_ids"
DOC_RECALL_BASIS = "doc_ids"
EVIDENCE_TEXT_RECALL_BASIS = "evidence_text"
INVALID_RECALL_BASIS = "invalid"
_WHITESPACE_PATTERN = re.compile(r"\s+", re.UNICODE)


class EvaluationServiceError(ValueError):
    """当评估输入格式错误时抛出。"""


class RetrievalServiceLike(Protocol):
    """EvaluationService 使用的最小检索边界。"""

    def retrieve(
        self,
        *,
        root: Path,
        query: str,
        top_k: int | None = None,
        mode: str | None = None,
        runtime_overrides: Mapping[str, object | None] | None = None,
    ) -> RetrievalResults:
        """返回一个评估问题的检索结果。"""


@dataclass(frozen=True)
class EvalCase:
    """一个带标注的检索评估用例。"""

    case_id: str
    question: str
    k: int | None = None
    expected_doc_ids: tuple[str, ...] = ()
    expected_chunk_ids: tuple[str, ...] = ()
    expected_evidence_text: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "EvalCase":
        case_id = _required_text(record, "case_id")
        question = _required_text(record, "question")
        return cls(
            case_id=case_id,
            question=question,
            k=_optional_positive_int(record.get("k"), field_name="k"),
            expected_doc_ids=_string_tuple(
                record.get("expected_doc_ids", record.get("relevant_doc_ids", ()))
            ),
            expected_chunk_ids=_string_tuple(
                record.get("expected_chunk_ids", record.get("relevant_chunk_ids", ()))
            ),
            expected_evidence_text=_string_tuple(
                record.get("expected_evidence_text", record.get("evidence_text", ()))
            ),
        )

    @property
    def has_ground_truth(self) -> bool:
        return bool(
            self.expected_chunk_ids or self.expected_doc_ids or self.expected_evidence_text
        )


@dataclass(frozen=True)
class PerCaseEvaluationResult:
    """单个用例的结构化 Recall@k 结果。"""

    case_id: str
    question: str
    k: int
    recall_hit: bool
    recall_basis: str
    retrieved_chunk_ids: tuple[str, ...] = ()
    retrieved_doc_ids: tuple[str, ...] = ()
    matched_expected_items: tuple[str, ...] = ()
    retrieval_mode: str | None = None
    fallback_used: bool = False
    error_message: str | None = None
    recall_at_k: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "question": self.question,
            "k": self.k,
            "recall_hit": self.recall_hit,
            "recall_basis": self.recall_basis,
            "retrieved_chunk_ids": list(self.retrieved_chunk_ids),
            "retrieved_doc_ids": list(self.retrieved_doc_ids),
            "matched_expected_items": list(self.matched_expected_items),
            "retrieval_mode": self.retrieval_mode,
            "fallback_used": self.fallback_used,
            "error_message": self.error_message,
            "recall_at_k": self.recall_at_k,
        }


@dataclass(frozen=True)
class EvaluationResult:
    """聚合后的检索评估结果。"""

    case_count: int
    recall_at_k: float
    failed_case_count: int
    per_case_results: tuple[PerCaseEvaluationResult, ...]
    evaluated_case_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_count": self.case_count,
            "evaluated_case_count": self.evaluated_case_count,
            "recall_at_k": self.recall_at_k,
            "failed_case_count": self.failed_case_count,
            "per_case_results": [result.to_dict() for result in self.per_case_results],
        }


@dataclass(frozen=True)
class _CaseScore:
    recall_hit: bool
    recall_basis: str
    matched_expected_items: tuple[str, ...]
    recall_at_k: float


class EvaluationService:
    """根据带标注评估用例计算仅检索 Recall@k。"""

    def __init__(
        self,
        *,
        retrieval_service: RetrievalServiceLike | None = None,
        metrics_collector_factory: Callable[[Path], MetricsCollector] | None = None,
    ) -> None:
        self._retrieval = retrieval_service or RetrievalService()
        self._metrics_collector_factory = metrics_collector_factory or MetricsCollector.from_root

    def load_cases(self, eval_cases_path: Path) -> tuple[EvalCase, ...]:
        """从对象结构 {schema_version, cases} 或顶层列表加载评估用例。"""

        payload = json.loads(eval_cases_path.read_text(encoding="utf-8"))
        records = payload.get("cases") if isinstance(payload, Mapping) else payload
        if not isinstance(records, list):
            raise EvaluationServiceError("eval_cases.json must contain a cases list.")
        return tuple(EvalCase.from_dict(record) for record in records)

    def evaluate_file(
        self,
        *,
        root: Path,
        eval_cases_path: Path,
        retrieval_mode: str | None = None,
        top_k: int | None = None,
        runtime_overrides: Mapping[str, object | None] | None = None,
    ) -> EvaluationResult:
        """加载评估用例并运行仅检索 Recall@k 评估。"""

        return self.evaluate_cases(
            root=root,
            cases=self.load_cases(eval_cases_path),
            retrieval_mode=retrieval_mode,
            top_k=top_k,
            runtime_overrides=runtime_overrides,
        )

    def evaluate_cases(
        self,
        *,
        root: Path,
        cases: tuple[EvalCase, ...],
        retrieval_mode: str | None = None,
        top_k: int | None = None,
        runtime_overrides: Mapping[str, object | None] | None = None,
    ) -> EvaluationResult:
        """使用 RetrievalService.retrieve 评估用例；不调用 LLM。"""

        metrics = self._metrics_collector_factory(root)
        started_at = _utc_now()
        start = perf_counter()
        per_case_results: list[PerCaseEvaluationResult] = []
        for case in cases:
            per_case_results.append(
                self._evaluate_case(
                    root=root,
                    case=case,
                    retrieval_mode=retrieval_mode,
                    top_k=top_k,
                    runtime_overrides=runtime_overrides,
                )
            )
        evaluated_results = tuple(
            result for result in per_case_results if result.recall_basis != INVALID_RECALL_BASIS
        )
        failed_case_count = sum(1 for result in per_case_results if result.error_message)
        recall_at_k = _macro_average(result.recall_at_k for result in evaluated_results)
        aggregate = EvaluationResult(
            case_count=len(cases),
            evaluated_case_count=len(evaluated_results),
            failed_case_count=failed_case_count,
            recall_at_k=recall_at_k,
            per_case_results=tuple(per_case_results),
        )
        ended_at = _utc_now()
        duration_ms = _elapsed_ms(start)
        metrics.record_success(
            event_type="evaluation",
            operation="recall_at_k",
            started_at=started_at,
            ended_at=ended_at,
            duration_ms=duration_ms,
            metadata=_evaluation_metrics_metadata(
                root=root,
                result=aggregate,
                retrieval_mode=retrieval_mode,
                top_k=top_k,
                status="success",
            ),
        )
        metrics.record_success(
            event_type="evaluation",
            operation="eval_total_time",
            started_at=started_at,
            ended_at=ended_at,
            duration_ms=duration_ms,
            metadata=_evaluation_metrics_metadata(
                root=root,
                result=aggregate,
                retrieval_mode=retrieval_mode,
                top_k=top_k,
                status="success",
            ),
        )
        return aggregate

    def _evaluate_case(
        self,
        *,
        root: Path,
        case: EvalCase,
        retrieval_mode: str | None,
        top_k: int | None,
        runtime_overrides: Mapping[str, object | None] | None,
    ) -> PerCaseEvaluationResult:
        case_k = case.k or top_k or DEFAULT_TOP_K
        if not case.has_ground_truth:
            return PerCaseEvaluationResult(
                case_id=case.case_id,
                question=case.question,
                k=case_k,
                recall_hit=False,
                recall_basis=INVALID_RECALL_BASIS,
                error_message="Evaluation case has no ground truth.",
            )
        try:
            retrieval_result = self._retrieval.retrieve(
                root=root,
                query=case.question,
                mode=retrieval_mode,
                top_k=case_k,
                runtime_overrides=runtime_overrides,
            )
        except Exception as exc:
            return PerCaseEvaluationResult(
                case_id=case.case_id,
                question=case.question,
                k=case_k,
                recall_hit=False,
                recall_basis=_recall_basis(case),
                retrieval_mode=retrieval_mode,
                fallback_used=False,
                error_message=str(exc),
                recall_at_k=0.0,
            )

        results = tuple(retrieval_result.results[:case_k])
        score = _score_case(case, results)
        return PerCaseEvaluationResult(
            case_id=case.case_id,
            question=case.question,
            k=case_k,
            recall_hit=score.recall_hit,
            recall_basis=score.recall_basis,
            retrieved_chunk_ids=tuple(str(result.chunk_id) for result in results),
            retrieved_doc_ids=tuple(str(result.source_doc_id) for result in results),
            matched_expected_items=score.matched_expected_items,
            retrieval_mode=retrieval_result.retrieval_mode,
            fallback_used=retrieval_result.fallback_used,
            error_message=None,
            recall_at_k=score.recall_at_k,
        )


def _score_case(case: EvalCase, results: tuple[Any, ...]) -> _CaseScore:
    if case.expected_chunk_ids:
        return _score_expected_items(
            expected=case.expected_chunk_ids,
            retrieved=tuple(str(result.chunk_id) for result in results),
            basis=CHUNK_RECALL_BASIS,
        )
    if case.expected_doc_ids:
        return _score_expected_items(
            expected=case.expected_doc_ids,
            retrieved=tuple(str(result.source_doc_id) for result in results),
            basis=DOC_RECALL_BASIS,
        )
    return _score_evidence_text(case.expected_evidence_text, results)


def _score_expected_items(
    *,
    expected: tuple[str, ...],
    retrieved: tuple[str, ...],
    basis: str,
) -> _CaseScore:
    retrieved_set = set(retrieved)
    matched = tuple(item for item in expected if item in retrieved_set)
    recall = len(matched) / len(expected) if expected else 0.0
    return _CaseScore(
        recall_hit=bool(matched),
        recall_basis=basis,
        matched_expected_items=matched,
        recall_at_k=round(recall, 6),
    )


def _score_evidence_text(expected_texts: tuple[str, ...], results: tuple[Any, ...]) -> _CaseScore:
    retrieved_text = "\n".join(str(result.text) for result in results)
    normalized_retrieved = _normalize_text(retrieved_text)
    matched = tuple(
        text
        for text in expected_texts
        if _normalize_text(text) and _normalize_text(text) in normalized_retrieved
    )
    recall = len(matched) / len(expected_texts) if expected_texts else 0.0
    return _CaseScore(
        recall_hit=bool(matched),
        recall_basis=EVIDENCE_TEXT_RECALL_BASIS,
        matched_expected_items=matched,
        recall_at_k=round(recall, 6),
    )


def _recall_basis(case: EvalCase) -> str:
    if case.expected_chunk_ids:
        return CHUNK_RECALL_BASIS
    if case.expected_doc_ids:
        return DOC_RECALL_BASIS
    if case.expected_evidence_text:
        return EVIDENCE_TEXT_RECALL_BASIS
    return INVALID_RECALL_BASIS


def _evaluation_metrics_metadata(
    *,
    root: Path,
    result: EvaluationResult,
    retrieval_mode: str | None,
    top_k: int | None,
    status: str,
) -> dict[str, Any]:
    return {
        "root": str(root.expanduser().resolve()),
        "retrieval_mode": retrieval_mode,
        "top_k": top_k,
        "case_count": result.case_count,
        "evaluated_case_count": result.evaluated_case_count,
        "failed_case_count": result.failed_case_count,
        "recall_at_k": result.recall_at_k,
        "status": status,
    }


def _required_text(record: Mapping[str, Any], field_name: str) -> str:
    value = str(record.get(field_name, "")).strip()
    if not value:
        raise EvaluationServiceError(f"Evaluation case is missing required field: {field_name}.")
    return value


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        items = (value,)
    elif isinstance(value, (list, tuple)):
        items = tuple(value)
    else:
        raise EvaluationServiceError("Expected ground truth field to be a string or list of strings.")
    return tuple(str(item).strip() for item in items if str(item).strip())


def _optional_positive_int(value: Any, *, field_name: str) -> int | None:
    if value is None:
        return None
    checked = int(value)
    if checked <= 0:
        raise EvaluationServiceError(f"{field_name} must be a positive integer.")
    return checked


def _normalize_text(text: str) -> str:
    return _WHITESPACE_PATTERN.sub(" ", str(text).strip().casefold())


def _macro_average(values: Iterable[float]) -> float:
    collected = tuple(float(value) for value in values)
    if not collected:
        return 0.0
    return round(sum(collected) / len(collected), 6)


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _elapsed_ms(start: float) -> float:
    return max(0.0, round((perf_counter() - start) * 1000, 3))
