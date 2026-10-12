"""Gradio web app: Ask the manual, Generate G-code, Validate G-code (SPEC 8).

Run: .venv/Scripts/python.exe app.py  ->  http://127.0.0.1:7860
Works without an LLM key (retrieval-only answers; generation disabled). Missing corpus/index data is rebuilt at startup.
"""
from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import Any

import gradio as gr

from gcoderag.generator import GcodeGenerator
from gcoderag.rag import RagPipeline
from gcoderag.validator import DIALECTS, GcodeValidator, ValidationReport

logger = logging.getLogger("app")

SAFETY_SENTENCE = ("Generated G-code is for learning and must be checked in a simulator (such as NC Viewer) and "
                   "reviewed by a qualified machinist before it is run on a real machine.")
NO_LLM_BANNER = "No LLM key configured: showing retrieved passages only"
NO_LLM_GENERATE = ("Generation needs an LLM. Set `LLM_PROVIDER` and `LLM_API_KEY` (see `.env.example`) and restart "
                   "the app. The **Validate G-code** tab works without a key.")
CITATION_HEADERS = ["#", "title", "URL"]
ISSUE_HEADERS = ["rule", "severity", "line", "message", "block"]
EXAMPLE_PROGRAM = """(dialect: linuxcnc) (the G1 moves have no F word: press Validate to see E005)
G21 G90 G17
T1 M6
S10000 M3
G0 Z5
G1 Z-1
G1 X20 Y0
G0 Z5
M5
M30
"""

ValidatorFactory = Callable[[str], GcodeValidator]


def ask_handler(pipeline: RagPipeline, question: str) -> tuple[str, list[list[Any]], str]:
    """Answer markdown, citation rows (#, title, URL), and the retrieved passages as markdown."""
    if not question.strip():
        return "Type a question first.", [], ""
    try:
        answer = pipeline.answer(question.strip())
    except Exception as exc:  # noqa: BLE001 - show provider/network errors in the UI instead of a stack trace
        logger.exception("ask failed")
        return f"Something went wrong while answering: `{type(exc).__name__}`. Try again.", [], ""
    numbers = {id(p): i for i, p in enumerate(answer.passages, start=1)}
    rows = [[numbers.get(id(h), "?"), h.title, h.url] for h in answer.citations]
    passages = "\n\n---\n\n".join(f"**[{i}] {p.title}**  \n{p.url}\n\n{p.text}"
                                  for i, p in enumerate(answer.passages, start=1))
    text = answer.text if answer.mode == "retrieval-only" else _answer_markdown(answer.text, answer.grounded)
    return text, rows, passages


def _answer_markdown(text: str, grounded: bool) -> str:
    if grounded:
        return text
    return f"{text}\n\n*The answer is not grounded in the retrieved documentation; check the passages below.*"


def report_markdown(report: ValidationReport) -> str:
    """One status line for a validation report."""
    status = "VALID" if report.ok else "INVALID"
    return (f"**{status}** ({report.dialect}): {len(report.errors)} error(s), {len(report.warnings)} warning(s), "
            f"{report.blocks} block(s)")


def issue_rows(report: ValidationReport) -> list[list[Any]]:
    """Issue table rows: rule, severity, line, message, block."""
    return [[i.rule, i.severity, i.line_no, i.message, i.raw] for i in report.errors + report.warnings]


def validate_handler(factory: ValidatorFactory, program: str, dialect: str) -> tuple[str, list[list[Any]]]:
    """Validate pasted G-code."""
    if not program.strip():
        return "Paste a program first.", []
    report = factory(dialect).validate(program)
    return report_markdown(report), issue_rows(report)


def generate_handler(generator: GcodeGenerator | None, request: str,
                     dialect: str) -> tuple[str, str, list[list[Any]], str]:
    """Program text, report status, issue rows, attempts line."""
    if generator is None:
        return "", NO_LLM_GENERATE, [], ""
    if not request.strip():
        return "", "Describe the part or operation first.", [], ""
    try:
        result = generator.generate(request.strip(), dialect=dialect)  # type: ignore[arg-type]
    except Exception as exc:  # noqa: BLE001 - LLM/API errors become a friendly message
        logger.exception("generate failed")
        return "", f"Generation failed: `{type(exc).__name__}`. Try again or check the LLM settings.", [], ""
    attempts = f"Attempts: {result.attempts} (validated after every attempt; max 3)"
    return result.program, report_markdown(result.report), issue_rows(result.report), attempts


def _ask_tab(pipeline: RagPipeline) -> None:
    if pipeline.llm is None:
        gr.Markdown(f"> **{NO_LLM_BANNER}**")
    question = gr.Textbox(label="Question", placeholder="How do I pause the program for 2 seconds?", lines=2)
    ask_btn = gr.Button("Ask", variant="primary")
    answer = gr.Markdown()
    citations = gr.Dataframe(headers=CITATION_HEADERS, label="Citations", interactive=False, wrap=True)
    with gr.Accordion("Retrieved passages", open=False):
        passages = gr.Markdown()
    outputs = [answer, citations, passages]
    ask_btn.click(lambda q: ask_handler(pipeline, q), inputs=question, outputs=outputs, api_name="ask")
    question.submit(lambda q: ask_handler(pipeline, q), inputs=question, outputs=outputs, api_name=False)


def _generate_tab(generator: GcodeGenerator | None) -> None:
    enabled = generator is not None
    if not enabled:
        gr.Markdown(f"> {NO_LLM_GENERATE}")
    request = gr.Textbox(label="Request", lines=3, interactive=enabled,
                         placeholder="Mill a 40 x 40 mm square pocket 3 mm deep with a 6 mm end mill")
    dialect = gr.Radio(list(DIALECTS), value="linuxcnc", label="Dialect", interactive=enabled)
    gen_btn = gr.Button("Generate", variant="primary", interactive=enabled)
    program = gr.Code(label="Program", language=None, interactive=False)
    status = gr.Markdown()
    issues = gr.Dataframe(headers=ISSUE_HEADERS, label="Validation report", interactive=False, wrap=True)
    attempts = gr.Markdown()
    gen_btn.click(lambda r, d: generate_handler(generator, r, d), inputs=[request, dialect],
                  outputs=[program, status, issues, attempts], api_name="generate")


def _validate_tab(factory: ValidatorFactory) -> None:
    program = gr.Code(label="G-code", language=None, value=EXAMPLE_PROGRAM, interactive=True, lines=14)
    dialect = gr.Radio(list(DIALECTS), value="linuxcnc", label="Dialect")
    val_btn = gr.Button("Validate", variant="primary")
    status = gr.Markdown()
    issues = gr.Dataframe(headers=ISSUE_HEADERS, label="Report", interactive=False, wrap=True)
    val_btn.click(lambda p, d: validate_handler(factory, p, d), inputs=[program, dialect], outputs=[status, issues],
                  api_name="validate")


def build_app(pipeline: RagPipeline, generator: GcodeGenerator | None, validator_factory: ValidatorFactory) -> gr.Blocks:
    """Assemble the three-tab Blocks UI (does not launch it)."""
    with gr.Blocks(title="CNC G-Code Assistant") as demo:
        gr.Markdown("# CNC G-Code Assistant\nAnswers from the LinuxCNC G-code documentation with citations, "
                    "LLM-generated programs checked by a deterministic validator, and a validator you can paste into.")
        with gr.Tab("Ask the manual"):
            _ask_tab(pipeline)
        with gr.Tab("Generate G-code"):
            _generate_tab(generator)
        with gr.Tab("Validate G-code"):
            _validate_tab(validator_factory)
        gr.Markdown(f"---\n**Safety:** {SAFETY_SENTENCE}")
    return demo


def ensure_data() -> None:
    """Build the corpus if `data/corpus/chunks.jsonl` is missing (a fresh clone or a Hugging Face Space)."""
    from gcoderag.config import get_paths
    from gcoderag.ingest.build import build_corpus

    paths = get_paths()
    if not (paths.corpus / "chunks.jsonl").exists():
        logger.info("corpus missing: building it (downloads the LinuxCNC docs)")
        stats = build_corpus(paths, skip_haas=os.getenv("SKIP_HAAS") == "1")
        logger.info("corpus built: %s docs, %s chunks", stats["num_docs"], stats["num_chunks"])


def main() -> None:
    """Load (or build) data, wire the pipeline, and launch the app."""
    from gcoderag.config import load_settings
    from gcoderag.index import load_retriever
    from gcoderag.llm import get_llm_from_env

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    load_settings()
    ensure_data()
    retriever = load_retriever()          # builds and saves data/index/ when it is missing
    llm = get_llm_from_env()
    logger.info("LLM: %s", llm.name if llm else "none (retrieval-only mode)")
    generator = GcodeGenerator(retriever, llm, GcodeValidator()) if llm else None
    build_app(RagPipeline(retriever, llm), generator, GcodeValidator).launch()


if __name__ == "__main__":
    main()
