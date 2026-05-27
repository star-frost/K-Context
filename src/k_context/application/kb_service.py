"""本地知识库生命周期操作的应用服务。"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Callable, Mapping
from uuid import uuid4

from k_context.application.answer_service import AnswerSource, GroundedAnswer, GroundedAnswerService
from k_context.application.ask_mcp_tool_loop import AskMCPToolLoop
from k_context.application.chunking_service import ChunkingService
from k_context.application.cleaning_service import CleaningService
from k_context.application.config_service import ConfigService
from k_context.application.deeprag_service import DEEPRAG_RAG_METHOD, DeepRAGService, normalize_rag_method
from k_context.application.document_parser import DocumentParser
from k_context.application.llm_client import (
    LLMClient,
    LLMClientError,
    LLMClientUnavailableError,
)
from k_context.application.mcp_client_bridge import MCPClientBridge
from k_context.application.metrics_collector import MetricsCollector
from k_context.application.prompt_builder import PromptBuilder, PromptBuilderError
from k_context.application.retrieval_service import DEFAULT_TOP_K, RetrievalService, SearchResult
from k_context.domain.models import Chunk, Document, DocumentBlock, KContextConfig, ParsedBlock
from k_context.infrastructure.storage.local_store import KbInitRecord, LocalKnowledgeBaseStore


_SENSITIVE_TEXT_PATTERNS = (
    re.compile(r"KCONTEXT_LLM_API_KEY\s*=\s*\S+", re.IGNORECASE),
    re.compile(r"\bKCONTEXT_LLM_API_KEY\b", re.IGNORECASE),
    re.compile(r"\bKCONTEXT_LLM_BASE_URL\b", re.IGNORECASE),
    re.compile(r"\bKCONTEXT_LLM_MODEL\b", re.IGNORECASE),
    re.compile(r"Authorization\s*:\s*Bearer\s+\S+", re.IGNORECASE),
    re.compile(r"Bearer\s+[A-Za-z0-9._~+/=-]{8,}", re.IGNORECASE),
    re.compile(r"(api[_-]?key|apikey|secret)\s*[:=]\s*\S+", re.IGNORECASE),
    re.compile(r"sk-[A-Za-z0-9._-]+", re.IGNORECASE),
)
_MCP_TIMEOUT_SECONDS_ENV = "KCONTEXT_MCP_TIMEOUT_SECONDS"


@dataclass(frozen=True)
class KbInitResult:
    """确认本地知识库存在后返回的结果。"""

    kb_root: Path
    created_paths: tuple[Path, ...]
    already_initialized: bool


@dataclass(frozen=True)
class KbAddResult:
    """添加并解析一个本地文档后返回的结果。"""

    document: Document
    blocks: tuple[DocumentBlock, ...]
    metadata_path: Path
    blocks_path: Path


@dataclass(frozen=True)
class ChunkingResult:
    """从持久化 DocumentBlock 记录生成切块后返回的结果。"""

    chunks: tuple[Chunk, ...]
    chunks_path: Path
    chunking_strategy: str


@dataclass(frozen=True)
class CleaningResult:
    """生成清洗后的 DocumentBlock 记录后返回的结果。"""

    blocks: tuple[DocumentBlock, ...]
    cleaned_blocks_path: Path
    cleaning_profile: str


@dataclass(frozen=True)
class SearchResults:
    """搜索持久化切块后返回的结果。"""

    results: tuple[SearchResult, ...]
    chunks_path: Path
    chunks_available: int


@dataclass(frozen=True)
class _RetrievalSummary:
    retrieval_mode: str
    requested_mode: str
    top_k: int


class KnowledgeBaseService:
    """协调知识库操作，同时不向 CLI 暴露存储细节。"""

    def __init__(
        self,
        store: LocalKnowledgeBaseStore | None = None,
        parser: DocumentParser | None = None,
        chunking: ChunkingService | None = None,
        cleaning: CleaningService | None = None,
        config_service: ConfigService | None = None,
        retrieval: RetrievalService | None = None,
        answer_service: GroundedAnswerService | None = None,
        prompt_builder: PromptBuilder | None = None,
        llm_client: LLMClient | None = None,
        llm_client_factory: Callable[[Path, KContextConfig], LLMClient] | None = None,
        mcp_bridge_factory: Callable[[Path], MCPClientBridge] | None = None,
        metrics_collector_factory: Callable[[Path], MetricsCollector] | None = None,
        deeprag_service: DeepRAGService | None = None,
    ) -> None:
        self._store = store or LocalKnowledgeBaseStore()
        self._parser = parser or DocumentParser()
        self._chunking = chunking or ChunkingService()
        self._cleaning = cleaning or CleaningService()
        self._config_service = config_service or ConfigService()
        self._retrieval = retrieval or RetrievalService()
        self._answer_service = answer_service or GroundedAnswerService()
        self._prompt_builder = prompt_builder or PromptBuilder()
        self._llm_client = llm_client
        self._llm_client_factory = llm_client_factory or self._default_llm_client_factory
        self._mcp_bridge_factory = mcp_bridge_factory or self._default_mcp_bridge_factory
        self._metrics_collector_factory = metrics_collector_factory or MetricsCollector.from_root
        self._deeprag = deeprag_service or DeepRAGService(
            retrieval_service=self._retrieval,
            answer_service=self._answer_service,
            prompt_builder=self._prompt_builder,
        )

    def init(self, root: Path) -> KbInitResult:
        """在 ``root`` 下创建本地知识库目录结构。"""

        record: KbInitRecord = self._store.initialize(root)
        return KbInitResult(
            kb_root=record.kb_root,
            created_paths=record.created_paths,
            already_initialized=record.already_initialized,
        )

    def add(self, root: Path, file_path: Path) -> KbAddResult:
        """注册文档，将其解析为 IR，并持久化元数据和块。"""

        kb_paths = self._store.require_initialized(root)
        parsed_document = self._parser.parse(file_path)
        document = Document.create(
            file_name=parsed_document.file_path.name,
            file_type=parsed_document.file_type,
            storage_ref=str(parsed_document.file_path),
            status="已解析",
            error_message=None,
        )
        blocks = tuple(
            self._to_document_block(document=document, parsed_block=block)
            for block in parsed_document.blocks
        )

        self._store.append_record(kb_paths.metadata_path, document.to_dict())
        self._store.append_records(kb_paths.blocks_path, (block.to_dict() for block in blocks))

        return KbAddResult(
            document=document,
            blocks=blocks,
            metadata_path=kb_paths.metadata_path,
            blocks_path=kb_paths.blocks_path,
        )

    def generate_chunks(
        self,
        root: Path,
        chunking_strategy: str | None = None,
    ) -> ChunkingResult:
        """从持久化 DocumentBlock IR 生成切块，并替换 chunks.jsonl。"""

        kb_paths = self._store.require_initialized(root)
        cleaning_result = self.generate_cleaned_blocks(root)
        blocks = cleaning_result.blocks
        strategy_name = chunking_strategy or self._config_service.load(root).chunking_strategy
        chunks = self._chunking.generate(blocks, chunking_strategy=strategy_name)
        self._store.replace_records(kb_paths.chunks_path, (chunk.to_dict() for chunk in chunks))
        return ChunkingResult(
            chunks=chunks,
            chunks_path=kb_paths.chunks_path,
            chunking_strategy=strategy_name,
        )

    def generate_cleaned_blocks(
        self,
        root: Path,
        cleaning_profile: str | None = None,
    ) -> CleaningResult:
        """生成清洗后的 DocumentBlock 记录，并替换 cleaned_blocks.jsonl。"""

        kb_paths = self._store.require_initialized(root)
        profile = cleaning_profile or self._config_service.load(root).cleaning_profile
        block_records = self._store.read_block_records(kb_paths)
        blocks = tuple(DocumentBlock.from_dict(record) for record in block_records)
        cleaned_blocks = self._cleaning.clean(blocks, cleaning_profile=profile)
        self._store.replace_cleaned_block_records(
            kb_paths,
            (block.to_dict() for block in cleaned_blocks),
        )
        return CleaningResult(
            blocks=cleaned_blocks,
            cleaned_blocks_path=kb_paths.cleaned_blocks_path,
            cleaning_profile=profile,
        )

    def search(self, root: Path, query: str, top_k: int = DEFAULT_TOP_K) -> SearchResults:
        """使用轻量本地检索策略搜索持久化切块。"""

        kb_paths = self._store.require_initialized(root)
        chunk_records = self._store.read_records(kb_paths.chunks_path)
        chunks = tuple(Chunk.from_dict(record) for record in chunk_records)
        results = self._retrieval.search(chunks=chunks, query=query, top_k=top_k)
        return SearchResults(
            results=results,
            chunks_path=kb_paths.chunks_path,
            chunks_available=len(chunks),
        )

    def ask(
        self,
        root: Path,
        question: str,
        top_k: int | None = None,
        *,
        mode: str | None = None,
        rag_method: str | None = None,
        runtime_overrides: Mapping[str, object | None] | None = None,
        use_llm: bool = True,
    ) -> GroundedAnswer:
        """使用检索切块回答问题，并在可用时优先使用 LLM。"""

        metrics = self._metrics_collector_factory(root)
        ask_started_at = _utc_now()
        ask_start = perf_counter()
        overrides = dict(runtime_overrides or {})
        forced_fallback_reason: str | None = None
        mcp_fallback_answer: GroundedAnswer | None = None
        llm_client: LLMClient | None = None
        config = self._config_service.load(root, runtime_overrides=overrides)
        effective_rag_method = normalize_rag_method(
            rag_method or str(overrides.get("rag_method") or config.rag_method)
        )
        if effective_rag_method == DEEPRAG_RAG_METHOD:
            deeprag_use_llm = use_llm
            deeprag_llm_unavailable_reason: str | None = None
            if deeprag_use_llm:
                try:
                    llm_client = self._get_llm_client(root, config)
                except (LLMClientError, LLMClientUnavailableError) as exc:
                    deeprag_use_llm = False
                    deeprag_llm_unavailable_reason = _sanitize_fallback_reason(str(exc))
            deeprag_result = self._deeprag.run(
                root=root,
                question=question,
                top_k=top_k,
                mode=mode,
                runtime_overrides=overrides,
                use_llm=deeprag_use_llm,
                llm_client=llm_client,
            )
            answer = deeprag_result.answer
            if deeprag_llm_unavailable_reason and answer.fallback_used:
                answer = replace(answer, fallback_reason=deeprag_llm_unavailable_reason)
            retrieval_summary = _RetrievalSummary(
                retrieval_mode=deeprag_result.retrieval_mode,
                requested_mode=deeprag_result.requested_mode,
                top_k=deeprag_result.top_k,
            )
            if deeprag_use_llm and not answer.fallback_used:
                metrics.record_success(
                    event_type="llm_call",
                    operation="llm_latency",
                    started_at=ask_started_at,
                    ended_at=_utc_now(),
                    duration_ms=answer.latency_ms,
                    metadata=_ask_metrics_metadata(
                        root=root,
                        retrieval_result=retrieval_summary,
                        answer=answer,
                        fallback_used=False,
                        token_usage=answer.token_usage.to_dict(),
                        token_usage_source=answer.token_usage_source,
                        llm_model=getattr(llm_client, "model", None),
                        status="success",
                    ),
                )
            if answer.fallback_used or not answer.sources:
                metrics.record_success(
                    event_type="ask_fallback",
                    operation="grounded_synthesis",
                    started_at=ask_started_at,
                    ended_at=_utc_now(),
                    duration_ms=_elapsed_ms(ask_start),
                    metadata=_ask_metrics_metadata(
                        root=root,
                        retrieval_result=retrieval_summary,
                        answer=answer,
                        fallback_used=True,
                        fallback_reason=answer.fallback_reason,
                        token_usage=answer.token_usage.to_dict(),
                        token_usage_source=answer.token_usage_source,
                        llm_model=getattr(llm_client, "model", None),
                        status="success",
                    ),
                )
            metrics.record_success(
                event_type="token_usage",
                operation="llm_token_usage",
                started_at=ask_started_at,
                ended_at=_utc_now(),
                duration_ms=0.0,
                metadata=_ask_metrics_metadata(
                    root=root,
                    retrieval_result=retrieval_summary,
                    answer=answer,
                    fallback_used=answer.fallback_used,
                    fallback_reason=answer.fallback_reason,
                    token_usage=answer.token_usage.to_dict(),
                    token_usage_source=answer.token_usage_source,
                    llm_model=getattr(llm_client, "model", None),
                    status="success",
                ),
            )
            metrics.record_success(
                event_type="ask",
                operation="ask_total_time",
                started_at=ask_started_at,
                ended_at=_utc_now(),
                duration_ms=_elapsed_ms(ask_start),
                metadata=_ask_metrics_metadata(
                    root=root,
                    retrieval_result=retrieval_summary,
                    answer=answer,
                    fallback_used=answer.fallback_used,
                    fallback_reason=answer.fallback_reason,
                    token_usage=answer.token_usage.to_dict(),
                    token_usage_source=answer.token_usage_source,
                    llm_model=getattr(llm_client, "model", None),
                    status="success",
                ),
            )
            self._append_session_record(
                root=root,
                question=question,
                answer=answer,
                retrieval_result=retrieval_summary,
                metrics=metrics,
            )
            return answer
        if use_llm:
            try:
                llm_client = self._get_llm_client(root, config)
                mcp_answer = AskMCPToolLoop(
                    llm_client=llm_client,
                    mcp_bridge=self._mcp_bridge_factory(root),
                ).run(
                    question=question,
                    top_k=top_k,
                    retrieval_mode=mode or str(overrides.get("retrieval_mode") or config.retrieval_mode),
                )
            except (LLMClientError, LLMClientUnavailableError) as exc:
                mcp_answer = GroundedAnswer(
                    answer="",
                    evidence_level="证据不足",
                    sources=(),
                    retrieval_mode=mode or str(overrides.get("retrieval_mode") or config.retrieval_mode),
                    top_k=top_k,
                    fallback_used=True,
                    fallback_reason=_sanitize_fallback_reason(str(exc)),
                )
            if not mcp_answer.fallback_used or mcp_answer.fallback_reason == "mcp_tool_result_empty":
                retrieval_summary = _RetrievalSummary(
                    retrieval_mode=mcp_answer.retrieval_mode,
                    requested_mode=mode or str(overrides.get("retrieval_mode") or config.retrieval_mode),
                    top_k=mcp_answer.top_k or top_k or int(overrides.get("top_k") or config.top_k),
                )
                _record_mcp_metrics(
                    metrics=metrics,
                    root=root,
                    retrieval_result=retrieval_summary,
                    answer=mcp_answer,
                    llm_model=getattr(llm_client, "model", None),
                )
                if not mcp_answer.fallback_used:
                    metrics.record_success(
                        event_type="llm_call",
                        operation="llm_latency",
                        started_at=ask_started_at,
                        ended_at=_utc_now(),
                        duration_ms=mcp_answer.latency_ms,
                        metadata=_ask_metrics_metadata(
                            root=root,
                            retrieval_result=retrieval_summary,
                            answer=mcp_answer,
                            fallback_used=False,
                            token_usage=mcp_answer.token_usage.to_dict(),
                            token_usage_source=mcp_answer.token_usage_source,
                            llm_model=getattr(llm_client, "model", None),
                            status="success",
                        ),
                    )
                else:
                    metrics.record_success(
                        event_type="ask_fallback",
                        operation="grounded_synthesis",
                        started_at=ask_started_at,
                        ended_at=_utc_now(),
                        duration_ms=_elapsed_ms(ask_start),
                        metadata=_ask_metrics_metadata(
                            root=root,
                            retrieval_result=retrieval_summary,
                            answer=mcp_answer,
                            fallback_used=True,
                            fallback_reason=mcp_answer.fallback_reason,
                            token_usage=mcp_answer.token_usage.to_dict(),
                            token_usage_source=mcp_answer.token_usage_source,
                            llm_model=getattr(llm_client, "model", None),
                            status="success",
                        ),
                    )
                metrics.record_success(
                    event_type="token_usage",
                    operation="llm_token_usage",
                    started_at=ask_started_at,
                    ended_at=_utc_now(),
                    duration_ms=0.0,
                    metadata=_ask_metrics_metadata(
                        root=root,
                        retrieval_result=retrieval_summary,
                        answer=mcp_answer,
                        fallback_used=mcp_answer.fallback_used,
                        fallback_reason=mcp_answer.fallback_reason,
                        token_usage=mcp_answer.token_usage.to_dict(),
                        token_usage_source=mcp_answer.token_usage_source,
                        llm_model=getattr(llm_client, "model", None),
                        status="success",
                    ),
                )
                metrics.record_success(
                    event_type="ask",
                    operation="ask_total_time",
                    started_at=ask_started_at,
                    ended_at=_utc_now(),
                    duration_ms=_elapsed_ms(ask_start),
                    metadata=_ask_metrics_metadata(
                        root=root,
                        retrieval_result=retrieval_summary,
                        answer=mcp_answer,
                        fallback_used=mcp_answer.fallback_used,
                        fallback_reason=mcp_answer.fallback_reason,
                        token_usage=mcp_answer.token_usage.to_dict(),
                        token_usage_source=mcp_answer.token_usage_source,
                        llm_model=getattr(llm_client, "model", None),
                        status="success",
                    ),
                )
                self._append_session_record(
                    root=root,
                    question=question,
                    answer=mcp_answer,
                    retrieval_result=retrieval_summary,
                    metrics=metrics,
                )
                return mcp_answer
            forced_fallback_reason = _sanitize_fallback_reason(
                mcp_answer.fallback_reason or "mcp_tool_loop_failed"
            )
            mcp_fallback_answer = mcp_answer
            _record_mcp_metrics(
                metrics=metrics,
                root=root,
                retrieval_result=_RetrievalSummary(
                    retrieval_mode=mcp_answer.retrieval_mode,
                    requested_mode=mode or str(overrides.get("retrieval_mode") or config.retrieval_mode),
                    top_k=mcp_answer.top_k or top_k or int(overrides.get("top_k") or config.top_k),
                ),
                answer=mcp_answer,
                llm_model=getattr(llm_client, "model", None),
            )
            metrics.record_failure(
                event_type="llm_call",
                operation="llm_latency",
                error_message=forced_fallback_reason,
                started_at=ask_started_at,
                ended_at=_utc_now(),
                duration_ms=mcp_answer.latency_ms,
                metadata=_ask_metrics_metadata(
                    root=root,
                    retrieval_result=_RetrievalSummary(
                        retrieval_mode=mcp_answer.retrieval_mode,
                        requested_mode=mode or str(overrides.get("retrieval_mode") or config.retrieval_mode),
                        top_k=mcp_answer.top_k or top_k or int(overrides.get("top_k") or config.top_k),
                    ),
                    answer=mcp_answer,
                    fallback_used=True,
                    fallback_reason=forced_fallback_reason,
                    token_usage=mcp_answer.token_usage.to_dict(),
                    token_usage_source=mcp_answer.token_usage_source,
                    llm_model=getattr(llm_client, "model", None),
                    status="failure",
                ),
            )
            use_llm = False
        retrieval_result = self._retrieval.retrieve(
            root=root,
            query=question,
            mode=mode,
            top_k=top_k,
            runtime_overrides=overrides,
        )
        grounded_answer = self._answer_service.synthesize(
            question,
            retrieval_result.results,
            retrieval_mode=retrieval_result.retrieval_mode,
            top_k=retrieval_result.top_k,
            fallback_used=retrieval_result.fallback_used,
            fallback_reason=retrieval_result.fallback_reason,
        )
        answer = grounded_answer
        llm_started_at: str | None = None
        llm_start: float | None = None
        if use_llm and retrieval_result.results:
            try:
                llm_request = self._prompt_builder.build(
                    question=question,
                    sources=retrieval_result.results,
                )
                config = self._config_service.load(root, runtime_overrides=overrides)
                llm_client = self._get_llm_client(root, config)
                llm_started_at = _utc_now()
                llm_start = perf_counter()
                llm_response = llm_client.generate(llm_request)
                metrics.record_success(
                    event_type="llm_call",
                    operation="llm_latency",
                    started_at=llm_started_at,
                    ended_at=_utc_now(),
                    duration_ms=llm_response.latency_ms,
                    metadata=_ask_metrics_metadata(
                        root=root,
                        retrieval_result=retrieval_result,
                        answer=grounded_answer,
                        fallback_used=False,
                        token_usage=llm_response.token_usage.to_dict(),
                        token_usage_source=llm_response.token_usage_source,
                        llm_model=getattr(llm_client, "model", None),
                        status="success",
                    ),
                )
                answer = GroundedAnswer(
                    answer=llm_response.answer,
                    evidence_level=grounded_answer.evidence_level,
                    sources=grounded_answer.sources,
                    retrieval_mode=retrieval_result.retrieval_mode,
                    top_k=retrieval_result.top_k,
                    fallback_used=False,
                    fallback_reason=None,
                    token_usage=llm_response.token_usage,
                    token_usage_source=llm_response.token_usage_source,
                    latency_ms=llm_response.latency_ms,
                )
            except (LLMClientError, LLMClientUnavailableError) as exc:
                fallback_reason = _sanitize_fallback_reason(str(exc))
                answer = self._answer_service.synthesize(
                    question,
                    retrieval_result.results,
                    retrieval_mode=retrieval_result.retrieval_mode,
                    top_k=retrieval_result.top_k,
                    fallback_used=True,
                    fallback_reason=fallback_reason,
                )
                metrics.record_failure(
                    event_type="llm_call",
                    operation="llm_latency",
                    error_message=fallback_reason,
                    started_at=llm_started_at or _utc_now(),
                    ended_at=_utc_now(),
                    duration_ms=_elapsed_ms(llm_start or perf_counter()),
                    metadata=_ask_metrics_metadata(
                        root=root,
                        retrieval_result=retrieval_result,
                        answer=answer,
                        fallback_used=True,
                        fallback_reason=fallback_reason,
                        token_usage=answer.token_usage.to_dict(),
                        token_usage_source=answer.token_usage_source,
                        llm_model=getattr(llm_client, "model", None),
                        status="failure",
                    ),
                )
            except PromptBuilderError as exc:
                answer = self._answer_service.synthesize(
                    question,
                    retrieval_result.results,
                    retrieval_mode=retrieval_result.retrieval_mode,
                    top_k=retrieval_result.top_k,
                    fallback_used=True,
                    fallback_reason=_sanitize_fallback_reason(str(exc)),
                )
        elif not use_llm and retrieval_result.results:
            fallback_reason = forced_fallback_reason or retrieval_result.fallback_reason or "no_llm_requested"
            answer = self._answer_service.synthesize(
                question,
                retrieval_result.results,
                retrieval_mode=retrieval_result.retrieval_mode,
                top_k=retrieval_result.top_k,
                fallback_used=True,
                fallback_reason=fallback_reason,
            )
        if mcp_fallback_answer is not None and not answer.tool_loop_count:
            answer = _with_mcp_diagnostics(answer, mcp_fallback_answer)
        if answer.fallback_used or not retrieval_result.results:
            metrics.record_success(
                event_type="ask_fallback",
                operation="grounded_synthesis",
                started_at=ask_started_at,
                ended_at=_utc_now(),
                duration_ms=_elapsed_ms(ask_start),
                metadata=_ask_metrics_metadata(
                    root=root,
                    retrieval_result=retrieval_result,
                    answer=answer,
                    fallback_used=True,
                    fallback_reason=answer.fallback_reason,
                    token_usage=answer.token_usage.to_dict(),
                    token_usage_source=answer.token_usage_source,
                    status="success",
                ),
            )
        metrics.record_success(
            event_type="token_usage",
            operation="llm_token_usage",
            started_at=ask_started_at,
            ended_at=_utc_now(),
            duration_ms=0.0,
            metadata=_ask_metrics_metadata(
                root=root,
                retrieval_result=retrieval_result,
                answer=answer,
                fallback_used=answer.fallback_used,
                fallback_reason=answer.fallback_reason,
                token_usage=answer.token_usage.to_dict(),
                token_usage_source=answer.token_usage_source,
                llm_model=getattr(llm_client, "model", None),
                status="success",
            ),
        )
        metrics.record_success(
            event_type="ask",
            operation="ask_total_time",
            started_at=ask_started_at,
            ended_at=_utc_now(),
            duration_ms=_elapsed_ms(ask_start),
            metadata=_ask_metrics_metadata(
                root=root,
                retrieval_result=retrieval_result,
                answer=answer,
                fallback_used=answer.fallback_used,
                fallback_reason=answer.fallback_reason,
                token_usage=answer.token_usage.to_dict(),
                token_usage_source=answer.token_usage_source,
                llm_model=getattr(llm_client, "model", None),
                status="success",
            ),
        )
        self._append_session_record(
            root=root,
            question=question,
            answer=answer,
            retrieval_result=retrieval_result,
            metrics=metrics,
        )
        return answer

    def _get_llm_client(self, root: Path, config: KContextConfig) -> LLMClient:
        if self._llm_client is not None:
            return self._llm_client
        return self._llm_client_factory(root, config)

    def _default_llm_client_factory(self, root: Path, config: KContextConfig) -> LLMClient:
        del root
        from k_context.infrastructure.llm import OpenAICompatibleLLMClient

        return OpenAICompatibleLLMClient.from_config(config)

    def _default_mcp_bridge_factory(self, root: Path) -> MCPClientBridge:
        from k_context.infrastructure.mcp_client.stdio_client import StdioMCPClientBridge

        return StdioMCPClientBridge(
            root=root,
            timeout_seconds=_env_float(_MCP_TIMEOUT_SECONDS_ENV, default=300.0),
        )

    def _to_document_block(self, document: Document, parsed_block: ParsedBlock) -> DocumentBlock:
        return DocumentBlock.create(
            source_doc_id=document.doc_id,
            source_doc_name=document.file_name,
            page=parsed_block.page,
            order=parsed_block.order,
            block_type=parsed_block.block_type,
            heading_path=parsed_block.heading_path,
            text=parsed_block.text,
            bbox=parsed_block.bbox,
        )

    def _append_session_record(
        self,
        *,
        root: Path,
        question: str,
        answer: GroundedAnswer,
        retrieval_result: object,
        metrics: MetricsCollector,
    ) -> None:
        """向 sessions.jsonl 追加脱敏问答摘要，不混入指标事件。"""

        started_at = _utc_now()
        start = perf_counter()
        record = _session_record(
            question=question,
            answer=answer,
            retrieval_result=retrieval_result,
        )
        try:
            kb_paths = self._store.require_initialized(root)
            self._store.append_record(kb_paths.sessions_path, record)
        except OSError as exc:
            metrics.record_failure(
                event_type="ask",
                operation="session_write",
                error_message=_sanitize_sensitive_text(str(exc)),
                started_at=started_at,
                ended_at=_utc_now(),
                duration_ms=_elapsed_ms(start),
                metadata={
                    "retrieval_mode": record["retrieval_mode"],
                    "requested_mode": record["requested_mode"],
                    "top_k": record["top_k"],
                    "source_count": len(record["sources"]),
                    "evidence_level": record["evidence_level"],
                    "fallback_used": record["fallback_used"],
                    "status": "failure",
                },
            )


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _elapsed_ms(start: float) -> float:
    return max(0.0, round((perf_counter() - start) * 1000, 3))


def _ask_metrics_metadata(
    *,
    root: Path,
    retrieval_result: object,
    answer: GroundedAnswer,
    fallback_used: bool,
    token_usage: dict[str, object],
    token_usage_source: str,
    status: str,
    fallback_reason: str | None = None,
    llm_model: str | None = None,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "root": str(root.expanduser().resolve()),
        "retrieval_mode": str(getattr(retrieval_result, "retrieval_mode")),
        "requested_mode": str(getattr(retrieval_result, "requested_mode")),
        "top_k": int(getattr(retrieval_result, "top_k")),
        "source_count": len(answer.sources),
        "evidence_level": answer.evidence_level,
        "fallback_used": fallback_used,
        "rag_method": answer.rag_method,
        "deep_rag_step_count": len(answer.deep_rag_steps),
        "token_usage_source": token_usage_source,
        "token_usage": token_usage,
        "prompt_tokens": token_usage.get("prompt_tokens"),
        "completion_tokens": token_usage.get("completion_tokens"),
        "total_tokens": token_usage.get("total_tokens"),
        "status": status,
    }
    if fallback_reason:
        metadata["fallback_reason"] = _sanitize_fallback_reason(fallback_reason)
    if llm_model:
        metadata["llm_model"] = llm_model
    if answer.tool_calls_used:
        metadata["tool_calls_used"] = answer.tool_calls_used
        metadata["mcp_server_transport"] = answer.mcp_server_transport
        metadata["tool_name"] = answer.tool_name
        metadata["tool_loop_count"] = answer.tool_loop_count
    elif answer.tool_loop_count:
        metadata["tool_calls_used"] = answer.tool_calls_used
        metadata["mcp_server_transport"] = answer.mcp_server_transport
        metadata["tool_name"] = answer.tool_name
        metadata["tool_loop_count"] = answer.tool_loop_count
    return metadata


def _record_mcp_metrics(
    *,
    metrics: MetricsCollector,
    root: Path,
    retrieval_result: object,
    answer: GroundedAnswer,
    llm_model: str | None,
) -> None:
    events = answer.mcp_metrics.get("events") if isinstance(answer.mcp_metrics, Mapping) else None
    if not isinstance(events, list | tuple):
        return
    base_metadata = _ask_metrics_metadata(
        root=root,
        retrieval_result=retrieval_result,
        answer=answer,
        fallback_used=answer.fallback_used,
        fallback_reason=answer.fallback_reason,
        token_usage=answer.token_usage.to_dict(),
        token_usage_source=answer.token_usage_source,
        llm_model=llm_model,
        status="success" if not answer.fallback_used else "failure",
    )
    for raw_event in events:
        if not isinstance(raw_event, Mapping):
            continue
        event_type = str(raw_event.get("event_type", "")).strip()
        operation = str(raw_event.get("operation", "")).strip()
        status = str(raw_event.get("status", "success")).strip() or "success"
        if not event_type or not operation:
            continue
        event_metadata = raw_event.get("metadata") if isinstance(raw_event.get("metadata"), Mapping) else {}
        merged_metadata = {
            **base_metadata,
            **_sanitize_jsonish(event_metadata),
            "status": status,
        }
        metrics.record_event(
            event_type=event_type,
            operation=operation,
            status=status,
            duration_ms=float(raw_event.get("duration_ms", 0.0) or 0.0),
            metadata=merged_metadata,
        )


def _with_mcp_diagnostics(answer: GroundedAnswer, mcp_answer: GroundedAnswer) -> GroundedAnswer:
    return replace(
        answer,
        tool_calls_used=mcp_answer.tool_calls_used,
        mcp_server_transport=mcp_answer.mcp_server_transport,
        tool_name=mcp_answer.tool_name,
        tool_loop_count=mcp_answer.tool_loop_count,
        tool_calls=mcp_answer.tool_calls,
        tool_results_summary=mcp_answer.tool_results_summary,
        mcp_metrics=mcp_answer.mcp_metrics,
    )


def _session_record(
    *,
    question: str,
    answer: GroundedAnswer,
    retrieval_result: object,
) -> dict[str, object]:
    record: dict[str, object] = {
        "session_id": f"session_{uuid4().hex}",
        "question": _sanitize_sensitive_text(question),
        "answer": _sanitize_sensitive_text(answer.answer),
        "evidence_level": answer.evidence_level,
        "sources": [_session_source(source) for source in answer.sources],
        "retrieval_mode": answer.retrieval_mode,
        "requested_mode": str(getattr(retrieval_result, "requested_mode")),
        "top_k": answer.top_k,
        "fallback_used": answer.fallback_used,
        "token_usage": answer.token_usage.to_dict(),
        "token_usage_source": answer.token_usage_source,
        "latency_ms": max(0.0, float(answer.latency_ms)),
        "rag_method": answer.rag_method,
        "deep_rag_steps": _sanitize_jsonish(answer.deep_rag_steps),
        "tool_calls_used": answer.tool_calls_used,
        "mcp_server_transport": answer.mcp_server_transport,
        "tool_name": answer.tool_name,
        "tool_loop_count": answer.tool_loop_count,
        "tool_calls": _sanitize_jsonish(answer.tool_calls),
        "tool_results_summary": _sanitize_jsonish(answer.tool_results_summary),
        "created_at": _utc_now(),
    }
    if answer.fallback_reason:
        record["fallback_reason"] = _sanitize_fallback_reason(answer.fallback_reason)
    return record


def _session_source(source: AnswerSource) -> dict[str, object]:
    return {
        "chunk_id": source.chunk_id,
        "source_doc_id": source.source_doc_id,
        "source_doc_name": _sanitize_sensitive_text(source.source_doc_name),
        "score": source.score,
        "retrieval_mode": source.retrieval_mode,
        "block_ids": list(source.block_ids),
        "page_start": source.page_start,
        "page_end": source.page_end,
        "quote": _sanitize_sensitive_text(source.quote) if source.quote is not None else None,
        "metadata": dict(source.metadata),
    }


def _sanitize_sensitive_text(text: str) -> str:
    sanitized = str(text)
    for pattern in _SENSITIVE_TEXT_PATTERNS:
        sanitized = pattern.sub("[REDACTED]", sanitized)
    return sanitized


def _sanitize_fallback_reason(reason: str) -> str:
    return _sanitize_sensitive_text(reason.strip() or "LLM unavailable.")


def _sanitize_jsonish(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            _sanitize_sensitive_text(str(key)): _sanitize_jsonish(nested)
            for key, nested in value.items()
            if str(key).casefold() != "root"
        }
    if isinstance(value, tuple | list):
        return [_sanitize_jsonish(item) for item in value]
    if isinstance(value, str):
        return _sanitize_sensitive_text(value)
    return value


def _env_float(name: str, *, default: float) -> float:
    raw_value = os.environ.get(name)
    if raw_value is None or not raw_value.strip():
        return default
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number of seconds.") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value
