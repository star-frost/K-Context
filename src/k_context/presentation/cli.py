"""命令行入口。"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

from k_context.application.document_parser import DocumentParseError
from k_context.application.config_service import ConfigService
from k_context.application.evaluation_service import EvaluationService, EvaluationServiceError
from k_context.application.index_service import IndexService, IndexServiceError
from k_context.application.kb_service import KnowledgeBaseService
from k_context.application.retrieval_service import (
    DEFAULT_TOP_K,
    RetrievalService,
    RetrievalServiceError,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kb",
        description="Local knowledge-base assistant CLI.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    init_parser = subcommands.add_parser(
        "init",
        help="Initialize a local knowledge base.",
    )
    init_parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Workspace root where the .kcontext directory will be created.",
    )
    init_parser.set_defaults(handler=_handle_init)

    add_parser = subcommands.add_parser(
        "add",
        help="Add one local document to the initialized knowledge base.",
    )
    add_parser.add_argument(
        "file_path",
        type=Path,
        help="PDF, Markdown, or TXT file to add.",
    )
    add_parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Workspace root containing the .kcontext directory.",
    )
    add_parser.set_defaults(handler=_handle_add)

    index_parser = subcommands.add_parser(
        "index",
        help="Build the local RAG index from parsed document blocks.",
    )
    index_parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Workspace root containing the .kcontext directory.",
    )
    index_parser.add_argument(
        "--chunking-strategy",
        help="Temporarily override the configured chunking strategy for this index run.",
    )
    index_parser.add_argument(
        "--cleaning-profile",
        help="Temporarily override the configured cleaning profile for this index run.",
    )
    index_parser.add_argument(
        "--embedding-model",
        help="Temporarily override the configured embedding model for this index run.",
    )
    index_parser.add_argument(
        "--embedding-device",
        help="Temporarily override the configured embedding device for this index run.",
    )
    index_parser.add_argument(
        "--vector-store-type",
        help="Temporarily override the configured vector store type for this index run.",
    )
    index_parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild the current pipeline collection before upserting vectors.",
    )
    index_parser.set_defaults(handler=_handle_index)

    search_parser = subcommands.add_parser(
        "search",
        help="Search persisted chunks with the configured retrieval mode.",
    )
    search_parser.add_argument("query", help="Query text.")
    search_parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Workspace root containing the .kcontext directory.",
    )
    search_parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Maximum number of matching chunks to return; overrides config for this run.",
    )
    search_parser.add_argument(
        "--mode",
        choices=("vector", "keyword"),
        help="Retrieval mode for this run; defaults to .kcontext/config.json retrieval_mode.",
    )
    search_parser.add_argument(
        "--embedding-model",
        help="Temporarily override the configured embedding model for vector search.",
    )
    search_parser.add_argument(
        "--embedding-device",
        help="Temporarily override the configured embedding device for vector search.",
    )
    search_parser.add_argument(
        "--vector-store-type",
        help="Temporarily override the configured vector store type for vector search.",
    )
    search_parser.add_argument(
        "--chunking-strategy",
        help="Temporarily override the configured chunking strategy for vector search.",
    )
    search_parser.add_argument(
        "--cleaning-profile",
        help="Temporarily override the configured cleaning profile for vector search.",
    )
    search_parser.set_defaults(handler=_handle_search)

    ask_parser = subcommands.add_parser(
        "ask",
        help="Answer a question using retrieved local chunks.",
    )
    ask_parser.add_argument("question", help="Question text.")
    ask_parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Workspace root containing the .kcontext directory.",
    )
    ask_parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Maximum number of chunks to use as sources; overrides config for this run.",
    )
    ask_parser.add_argument(
        "--mode",
        choices=("vector", "keyword"),
        help="Retrieval mode for this run; defaults to .kcontext/config.json retrieval_mode.",
    )
    ask_parser.add_argument(
        "--rag-method",
        choices=("standard", "deeprag"),
        help="RAG orchestration method for this run; defaults to .kcontext/config.json rag_method.",
    )
    ask_parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Use grounded synthesis fallback without calling the configured LLM.",
    )
    ask_parser.set_defaults(handler=_handle_ask)

    eval_parser = subcommands.add_parser(
        "eval",
        help="Evaluate retrieval Recall@k from annotated eval cases.",
    )
    eval_parser.add_argument(
        "eval_cases_path",
        type=Path,
        help="JSON file containing annotated retrieval evaluation cases.",
    )
    eval_parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Workspace root containing the .kcontext directory.",
    )
    eval_parser.add_argument(
        "--mode",
        choices=("vector", "keyword"),
        help="Retrieval mode for this eval run; defaults to .kcontext/config.json retrieval_mode.",
    )
    eval_parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Default Top-K for eval cases without an explicit k value.",
    )
    eval_parser.add_argument(
        "--output-jsonl",
        type=Path,
        help="Optional path to write one JSON object per evaluated case.",
    )
    eval_parser.add_argument(
        "--with-answers",
        action="store_true",
        help="Also run the ask flow for each eval case and include answer metrics.",
    )
    eval_parser.add_argument(
        "--rag-method",
        choices=("standard", "deeprag"),
        help="Only with --with-answers: RAG orchestration method for answer generation.",
    )
    eval_parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Only with --with-answers: force grounded synthesis without calling the LLM.",
    )
    eval_parser.set_defaults(handler=_handle_eval)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    _configure_stdio_encoding()
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)


def _configure_stdio_encoding() -> None:
    """当输出包含当前代码页之外的字符时，避免 CLI 崩溃。

    CLI 可能打印包含当前 Windows 代码页之外 Unicode 字符的 LLM 回答
    和 PDF 检索片段。保留流的当前编码，使使用 ``text=True`` 的子进程
    调用方仍可按平台默认值解码输出；但用替换字符处理不可编码字符，
    避免成功命令在打印阶段崩溃。
    """

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if not callable(reconfigure):
            continue
        try:
            reconfigure(errors="replace")
        except (OSError, ValueError):
            continue


def _handle_init(args: argparse.Namespace) -> int:
    result = KnowledgeBaseService().init(args.root)

    status = "already initialized" if result.already_initialized else "initialized"
    print(f"Knowledge base {status}: {result.kb_root}")
    if result.created_paths:
        print("Created:")
        for path in result.created_paths:
            print(f"  - {path}")
    return 0


def _handle_add(args: argparse.Namespace) -> int:
    try:
        result = KnowledgeBaseService().add(args.root, args.file_path)
    except (DocumentParseError, FileNotFoundError, OSError) as exc:
        print(f"Error: {exc}")
        return 1

    print("Document added:")
    print(f"  document_id: {result.document.doc_id}")
    print(f"  file_name: {result.document.file_name}")
    print(f"  file_type: {result.document.file_type}")
    print(f"  metadata_path: {result.metadata_path}")
    print(f"  blocks_path: {result.blocks_path}")
    print(f"  block_count: {len(result.blocks)}")
    return 0


def _handle_index(args: argparse.Namespace) -> int:
    runtime_overrides = _index_runtime_overrides(args)
    try:
        result = IndexService().build(
            args.root,
            runtime_overrides=runtime_overrides,
            rebuild=args.rebuild,
        )
        config_service = ConfigService()
        effective_config = config_service.load(args.root, runtime_overrides=runtime_overrides)
        chroma_persist_dir = config_service.resolve_chroma_persist_dir(
            args.root,
            effective_config,
        )
    except (IndexServiceError, FileNotFoundError, OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("Index generated:")
    print(f"  cleaned_blocks_path: {result.cleaned_blocks_path}")
    print(f"  chunks_path: {result.chunks_path}")
    print(f"  chunk_count: {len(result.chunks)}")
    print(f"  vector_record_count: {len(result.vector_records)}")
    print(f"  vector_store_type: {effective_config.vector_store_type}")
    print(f"  chroma_persist_dir: {chroma_persist_dir}")
    print(f"  rebuild: {result.rebuild}")
    print(f"  metrics_path: {result.metrics_path}")
    return 0


def _index_runtime_overrides(args: argparse.Namespace) -> dict[str, object]:
    mapping = {
        "chunking_strategy": args.chunking_strategy,
        "cleaning_profile": args.cleaning_profile,
        "embedding_model": args.embedding_model,
        "embedding_device": args.embedding_device,
        "vector_store_type": args.vector_store_type,
    }
    return {key: value for key, value in mapping.items() if value is not None}


def _handle_search(args: argparse.Namespace) -> int:
    runtime_overrides = _search_runtime_overrides(args)
    try:
        result = RetrievalService().retrieve(
            root=args.root,
            query=args.query,
            mode=args.mode,
            top_k=args.top_k,
            runtime_overrides=runtime_overrides,
        )
    except (RetrievalServiceError, FileNotFoundError, OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if result.chunks_available == 0:
        print(f"No chunks available. Run `kb index --root {args.root}` before searching.")
        return 0
    if not result.results:
        print("No matching chunks found.")
        return 0

    print(f"Search results: {len(result.results)}")
    print(f"retrieval_mode: {result.retrieval_mode}")
    print(f"fallback_used: {result.fallback_used}")
    if result.fallback_reason:
        print(f"fallback_reason: {result.fallback_reason}")
    for position, search_result in enumerate(result.results, start=1):
        print(f"[{position}]")
        print(f"  chunk_id: {search_result.chunk_id}")
        print(f"  source_doc_id: {search_result.source_doc_id}")
        print(f"  source_doc_name: {search_result.source_doc_name}")
        print(f"  score: {search_result.score:g}")
        print(f"  retrieval_mode: {search_result.retrieval_mode}")
        print(f"  block_ids: {', '.join(search_result.block_ids)}")
        print(f"  text: {_search_result_snippet(search_result)}")
        fallback_used = search_result.metadata.get("fallback_used")
        fallback_reason = search_result.metadata.get("fallback_reason")
        if fallback_used is not None:
            print(f"  fallback_used: {fallback_used}")
        if fallback_reason:
            print(f"  fallback_reason: {fallback_reason}")
    return 0


def _search_runtime_overrides(args: argparse.Namespace) -> dict[str, object]:
    mapping = {
        "retrieval_mode": args.mode,
        "top_k": args.top_k,
        "embedding_model": args.embedding_model,
        "embedding_device": args.embedding_device,
        "vector_store_type": args.vector_store_type,
        "chunking_strategy": args.chunking_strategy,
        "cleaning_profile": args.cleaning_profile,
    }
    return {key: value for key, value in mapping.items() if value is not None}


def _search_result_snippet(search_result: object) -> str:
    snippet = getattr(search_result, "snippet", None)
    if callable(snippet):
        return str(snippet())
    text = str(getattr(search_result, "text"))
    return text if len(text) <= 160 else text[:160].rstrip() + "..."


def _handle_ask(args: argparse.Namespace) -> int:
    runtime_overrides = _ask_runtime_overrides(args)
    try:
        answer = KnowledgeBaseService().ask(
            args.root,
            args.question,
            args.top_k,
            mode=args.mode,
            rag_method=args.rag_method,
            runtime_overrides=runtime_overrides,
            use_llm=not args.no_llm,
        )
    except (RetrievalServiceError, FileNotFoundError, OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("answer:")
    print(answer.answer)
    print(f"evidence_level: {answer.evidence_level}")
    print(f"retrieval_mode: {answer.retrieval_mode}")
    print(f"rag_method: {answer.rag_method}")
    if answer.deep_rag_steps:
        print(f"deep_rag_step_count: {len(answer.deep_rag_steps)}")
    print(f"fallback_used: {answer.fallback_used}")
    if answer.fallback_reason:
        print(f"fallback_reason: {answer.fallback_reason}")
    print(f"token_usage: {answer.token_usage.to_dict()}")
    print(f"token_usage_source: {answer.token_usage_source}")
    print(f"latency_ms: {answer.latency_ms:g}")
    print(f"tool_calls_used: {answer.tool_calls_used}")
    print(f"mcp_server_transport: {answer.mcp_server_transport}")
    print(f"tool_name: {answer.tool_name}")
    print(f"tool_loop_count: {answer.tool_loop_count}")
    print("sources:")
    if not answer.sources:
        print("  []")
        return 0

    for source in answer.sources:
        print(f"  - chunk_id: {source.chunk_id}")
        print(f"    source_doc_id: {source.source_doc_id}")
        print(f"    source_doc_name: {source.source_doc_name}")
        print(f"    score: {source.score:g}")
        print(f"    retrieval_mode: {source.retrieval_mode}")
        print(f"    block_ids: {', '.join(source.block_ids)}")
        print(f"    page_start: {source.page_start}")
        print(f"    page_end: {source.page_end}")
        if source.quote:
            print(f"    quote: {source.quote}")
        if source.metadata:
            print(f"    metadata: {source.metadata}")
        if source.fallback_used:
            print(f"    fallback_used: {source.fallback_used}")
        if source.fallback_reason:
            print(f"    fallback_reason: {source.fallback_reason}")
    return 0


def _ask_runtime_overrides(args: argparse.Namespace) -> dict[str, object]:
    mapping = {
        "retrieval_mode": args.mode,
        "rag_method": args.rag_method,
        "top_k": args.top_k,
    }
    return {key: value for key, value in mapping.items() if value is not None}


def _handle_eval(args: argparse.Namespace) -> int:
    runtime_overrides = _eval_runtime_overrides(args)
    try:
        result = EvaluationService().evaluate_file(
            root=args.root,
            eval_cases_path=args.eval_cases_path,
            retrieval_mode=args.mode,
            top_k=args.top_k,
            runtime_overrides=runtime_overrides,
        )
        answer_records = (
            _eval_answer_records(args, result.per_case_results, runtime_overrides)
            if args.with_answers
            else {}
        )
        if args.output_jsonl:
            _write_eval_jsonl(
                args.root,
                args.output_jsonl,
                _eval_output_rows(result.per_case_results, answer_records),
            )
    except (
        EvaluationServiceError,
        FileNotFoundError,
        json.JSONDecodeError,
        OSError,
        ValueError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    invalid_case_count = result.case_count - result.evaluated_case_count
    print("Evaluation summary:")
    print(f"  case_count: {result.case_count}")
    print(f"  evaluated_case_count: {result.evaluated_case_count}")
    print(f"  failed_case_count: {result.failed_case_count}")
    print(f"  invalid_case_count: {invalid_case_count}")
    print(f"  recall@k: {result.recall_at_k:g}")
    print(f"  retrieval_mode: {args.mode or 'config'}")
    print(f"  top_k: {args.top_k if args.top_k is not None else 'case_or_config'}")
    if args.with_answers:
        answer_summary = _eval_answer_summary(answer_records)
        print("  with_answers: True")
        print(f"  rag_method: {args.rag_method or 'config'}")
        print(f"  no_llm: {args.no_llm}")
        print(f"  avg_e2e_latency_ms: {answer_summary['avg_e2e_latency_ms']:g}")
        print(f"  avg_llm_latency_ms: {answer_summary['avg_llm_latency_ms']:g}")
        print(f"  total_token_usage: {answer_summary['total_token_usage']}")
        print(f"  answer_failed_case_count: {answer_summary['answer_failed_case_count']}")
    if args.output_jsonl:
        print(f"  output_jsonl: {_resolve_output_jsonl_path(args.root, args.output_jsonl)}")
    return 0


def _eval_runtime_overrides(args: argparse.Namespace) -> dict[str, object]:
    mapping = {
        "retrieval_mode": args.mode,
        "top_k": args.top_k,
        "rag_method": args.rag_method,
    }
    return {key: value for key, value in mapping.items() if value is not None}


def _eval_answer_records(
    args: argparse.Namespace,
    per_case_results: object,
    runtime_overrides: Mapping[str, object],
) -> dict[str, dict[str, Any]]:
    kb_service = KnowledgeBaseService()
    records: dict[str, dict[str, Any]] = {}
    for case_result in per_case_results:
        case_id = str(getattr(case_result, "case_id"))
        question = str(getattr(case_result, "question"))
        case_k = int(getattr(case_result, "k"))
        started = perf_counter()
        try:
            answer = kb_service.ask(
                args.root,
                question,
                case_k,
                mode=args.mode,
                rag_method=args.rag_method,
                runtime_overrides=runtime_overrides,
                use_llm=not args.no_llm,
            )
            records[case_id] = {
                "answer": _sanitize_eval_text(answer.answer),
                "evidence_level": answer.evidence_level,
                "answer_sources": [_eval_answer_source(source) for source in answer.sources],
                "answer_latency_ms": max(0.0, float(answer.latency_ms)),
                "answer_e2e_latency_ms": _elapsed_ms(started),
                "token_usage": answer.token_usage.to_dict(),
                "token_usage_source": answer.token_usage_source,
                "answer_fallback_used": answer.fallback_used,
                "answer_fallback_reason": (
                    _sanitize_eval_text(answer.fallback_reason) if answer.fallback_reason else None
                ),
                "rag_method": answer.rag_method,
                "deep_rag_steps": _sanitize_eval_jsonish(answer.deep_rag_steps),
                "answer_error_message": None,
            }
        except Exception as exc:  # noqa: BLE001 - eval records per-case ask failures instead of aborting.
            error_message = _sanitize_eval_text(str(exc))
            records[case_id] = {
                "answer": None,
                "evidence_level": None,
                "answer_sources": [],
                "answer_latency_ms": 0.0,
                "answer_e2e_latency_ms": _elapsed_ms(started),
                "token_usage": _unavailable_token_usage(),
                "token_usage_source": "unavailable",
                "answer_fallback_used": True,
                "answer_fallback_reason": error_message,
                "answer_error_message": error_message,
            }
    return records


def _eval_answer_source(source: object) -> dict[str, Any]:
    return {
        "chunk_id": getattr(source, "chunk_id"),
        "source_doc_id": getattr(source, "source_doc_id"),
        "source_doc_name": _sanitize_eval_text(str(getattr(source, "source_doc_name"))),
        "score": getattr(source, "score"),
        "retrieval_mode": getattr(source, "retrieval_mode"),
        "block_ids": list(getattr(source, "block_ids")),
        "page_start": getattr(source, "page_start"),
        "page_end": getattr(source, "page_end"),
        "quote": (
            _sanitize_eval_text(str(getattr(source, "quote")))
            if getattr(source, "quote") is not None
            else None
        ),
    }


def _eval_answer_summary(answer_records: Mapping[str, Mapping[str, Any]]) -> dict[str, float | int]:
    e2e_latencies = [
        float(record["answer_e2e_latency_ms"])
        for record in answer_records.values()
        if record.get("answer_e2e_latency_ms") is not None
    ]
    llm_latencies = [
        float(record["answer_latency_ms"])
        for record in answer_records.values()
        if record.get("answer_latency_ms") is not None
    ]
    total_tokens = 0
    for record in answer_records.values():
        token_usage = record.get("token_usage")
        if isinstance(token_usage, Mapping) and isinstance(token_usage.get("total_tokens"), int):
            total_tokens += int(token_usage["total_tokens"])
    return {
        "avg_e2e_latency_ms": _average(e2e_latencies),
        "avg_llm_latency_ms": _average(llm_latencies),
        "total_token_usage": total_tokens,
        "answer_failed_case_count": sum(
            1 for record in answer_records.values() if record.get("answer_error_message")
        ),
    }


def _eval_output_rows(
    per_case_results: object,
    answer_records: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for case_result in per_case_results:
        row = case_result.to_dict()
        answer_record = answer_records.get(str(row["case_id"]))
        if answer_record is not None:
            row.update(answer_record)
        rows.append(row)
    return tuple(rows)


def _write_eval_jsonl(root: Path, output_path: Path, rows: object) -> None:
    resolved_path = _resolve_output_jsonl_path(root, output_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved_path.open("w", encoding="utf-8") as file:
        for row in rows:
            payload = row.to_dict() if hasattr(row, "to_dict") else dict(row)
            file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _resolve_output_jsonl_path(root: Path, output_path: Path) -> Path:
    if output_path.is_absolute():
        return output_path
    return root.expanduser().resolve() / output_path


def _unavailable_token_usage() -> dict[str, int | str | None]:
    return {
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "source": "unavailable",
    }


def _elapsed_ms(start: float) -> float:
    return max(0.0, round((perf_counter() - start) * 1000, 3))


def _average(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 3)


_EVAL_SENSITIVE_TEXT_PATTERNS = (
    re.compile(r"KCONTEXT_LLM_API_KEY\s*=\s*\S+", re.IGNORECASE),
    re.compile(r"\bKCONTEXT_LLM_API_KEY\b", re.IGNORECASE),
    re.compile(r"Authorization\s*:\s*Bearer\s+\S+", re.IGNORECASE),
    re.compile(r"Bearer\s+[A-Za-z0-9._~+/=-]{8,}", re.IGNORECASE),
    re.compile(r"(api[_-]?key|apikey|secret)\s*[:=]\s*\S+", re.IGNORECASE),
    re.compile(r"sk-[A-Za-z0-9._-]+", re.IGNORECASE),
)


def _sanitize_eval_text(text: str | None) -> str | None:
    if text is None:
        return None
    sanitized = str(text)
    for pattern in _EVAL_SENSITIVE_TEXT_PATTERNS:
        sanitized = pattern.sub("[REDACTED]", sanitized)
    return sanitized


def _sanitize_eval_jsonish(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _sanitize_eval_jsonish(nested) for key, nested in value.items()}
    if isinstance(value, tuple | list):
        return [_sanitize_eval_jsonish(item) for item in value]
    if isinstance(value, str):
        return _sanitize_eval_text(value)
    return value


if __name__ == "__main__":
    raise SystemExit(main())
