"""Command line interface: python -m gcoderag.cli {search,ask,generate,validate,docs}."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from gcoderag.index import MODES


def _cmd_search(args: argparse.Namespace) -> int:
    """Print rank, score, doc_id and title for the top-k chunks."""
    from gcoderag.index import load_retriever

    retriever = load_retriever()
    for rank, hit in enumerate(retriever.retrieve(args.query, k=args.k, mode=args.mode), start=1):
        print(f"{rank:>2}  {hit.score:.4f}  {hit.doc_id:<40}  {hit.title}")
    return 0


def _cmd_docs(_: argparse.Namespace) -> int:
    """List every document id and title in the corpus."""
    from gcoderag.config import get_paths
    from gcoderag.ingest import load_docs

    for doc in load_docs(get_paths().corpus / "docs.jsonl"):
        print(f"{doc.doc_id:<50} {doc.title}")
    return 0


def _cmd_ask(args: argparse.Namespace) -> int:
    """Answer a question with citations (LLM if configured, else retrieval-only)."""
    from gcoderag.index import load_retriever
    from gcoderag.llm import get_llm_from_env
    from gcoderag.rag import RagPipeline

    answer = RagPipeline(load_retriever(), get_llm_from_env()).answer(args.question)
    print(f"[mode: {answer.mode}, grounded: {answer.grounded}]\n{answer.text}\n")
    for i, hit in enumerate(answer.citations, start=1):
        print(f"[{i}] {hit.title} - {hit.url}")
    return 0


def _cmd_generate(args: argparse.Namespace) -> int:
    """Generate a validated G-code program (requires an LLM key)."""
    from gcoderag.generator import GcodeGenerator
    from gcoderag.index import load_retriever
    from gcoderag.llm import get_llm_from_env
    from gcoderag.validator import GcodeValidator

    llm = get_llm_from_env()
    if llm is None:
        print("No LLM key configured: generation needs LLM_PROVIDER and LLM_API_KEY in .env", file=sys.stderr)
        return 2
    result = GcodeGenerator(load_retriever(), llm, GcodeValidator(args.dialect)).generate(args.request, args.dialect)
    print(result.program)
    print(f"\n(attempts: {result.attempts})\n{result.report.summary()}")
    return 0 if result.report.ok else 1


def _cmd_validate(args: argparse.Namespace) -> int:
    """Validate a G-code file; exit 0 when it has no errors."""
    from gcoderag.validator import GcodeValidator

    report = GcodeValidator(args.dialect).validate(Path(args.file).read_text(encoding="utf-8"))
    print(json.dumps(report.to_dict(), indent=2) if args.json else report.summary())
    return 0 if report.ok else 1


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with its subcommands."""
    parser = argparse.ArgumentParser(prog="gcoderag", description="CNC G-code RAG assistant")
    sub = parser.add_subparsers(dest="command", required=True)
    search = sub.add_parser("search", help="retrieve chunks for a query")
    search.add_argument("query")
    search.add_argument("--mode", choices=MODES, default="hybrid")
    search.add_argument("--k", type=int, default=5)
    search.set_defaults(func=_cmd_search)
    sub.add_parser("docs", help="list corpus documents").set_defaults(func=_cmd_docs)
    ask = sub.add_parser("ask", help="answer a question with citations")
    ask.add_argument("question")
    ask.set_defaults(func=_cmd_ask)
    gen = sub.add_parser("generate", help="generate and validate a G-code program")
    gen.add_argument("request")
    gen.add_argument("--dialect", choices=("linuxcnc", "haas"), default="linuxcnc")
    gen.set_defaults(func=_cmd_generate)
    val = sub.add_parser("validate", help="validate a G-code file")
    val.add_argument("file")
    val.add_argument("--dialect", choices=("linuxcnc", "haas"), default="linuxcnc")
    val.add_argument("--json", action="store_true")
    val.set_defaults(func=_cmd_validate)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    from gcoderag.config import load_settings

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    load_settings()
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
