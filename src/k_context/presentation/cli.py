"""Command-line entry point for the first implementation slice."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from k_context.application.document_parser import DocumentParseError
from k_context.application.kb_service import KnowledgeBaseService
from k_context.application.retrieval_service import DEFAULT_TOP_K


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
        help="Generate chunks from parsed document blocks.",
    )
    index_parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Workspace root containing the .kcontext directory.",
    )
    index_parser.set_defaults(handler=_handle_index)

    search_parser = subcommands.add_parser(
        "search",
        help="Search persisted chunks with local keyword matching.",
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
        default=DEFAULT_TOP_K,
        help="Maximum number of matching chunks to return.",
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
        default=DEFAULT_TOP_K,
        help="Maximum number of chunks to use as sources.",
    )
    ask_parser.set_defaults(handler=_handle_ask)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)


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
    try:
        result = KnowledgeBaseService().generate_chunks(args.root)
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1

    print("Index generated:")
    print(f"  chunks_path: {result.chunks_path}")
    print(f"  chunk_count: {len(result.chunks)}")
    return 0


def _handle_search(args: argparse.Namespace) -> int:
    try:
        result = KnowledgeBaseService().search(args.root, args.query, args.top_k)
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1

    if result.chunks_available == 0:
        print(f"No chunks available. Run `kb index --root {args.root}` before searching.")
        return 0
    if not result.results:
        print("No matching chunks found.")
        return 0

    print(f"Search results: {len(result.results)}")
    for position, search_result in enumerate(result.results, start=1):
        chunk = search_result.chunk
        print(f"[{position}]")
        print(f"  chunk_id: {chunk.chunk_id}")
        print(f"  source_doc_id: {chunk.source_doc_id}")
        print(f"  source_doc_name: {chunk.source_doc_name}")
        print(f"  score: {search_result.score:g}")
        print(f"  block_ids: {', '.join(chunk.block_ids)}")
        print(f"  text: {search_result.snippet()}")
    return 0


def _handle_ask(args: argparse.Namespace) -> int:
    try:
        answer = KnowledgeBaseService().ask(args.root, args.question, args.top_k)
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1

    print("answer:")
    print(answer.answer)
    print(f"evidence_level: {answer.evidence_level}")
    print("sources:")
    if not answer.sources:
        print("  []")
        return 0

    for source in answer.sources:
        print(f"  - chunk_id: {source.chunk_id}")
        print(f"    source_doc_id: {source.source_doc_id}")
        print(f"    source_doc_name: {source.source_doc_name}")
        print(f"    score: {source.score:g}")
        print(f"    block_ids: {', '.join(source.block_ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
