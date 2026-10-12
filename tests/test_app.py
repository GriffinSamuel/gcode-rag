"""The Gradio app builds without launching, and its handlers work offline with FakeEmbedder/FakeLLM."""
from __future__ import annotations

from functools import partial

import gradio as gr

import app
from gcoderag.generator import GcodeGenerator
from gcoderag.llm import FakeLLM
from gcoderag.rag import RagPipeline
from gcoderag.validator import GcodeValidator

VALID = "```gcode\nG21 G90 G17\nT1 M6\nS9000 M3\nG0 Z5\nG1 Z-1 F100\nG0 Z5\nM5\nM30\n```"
factory = partial(GcodeValidator, use_corpus=False)


def test_blocks_build_with_and_without_llm(toy_retriever) -> None:
    llm = FakeLLM([VALID])
    with_llm = app.build_app(RagPipeline(toy_retriever, llm), GcodeGenerator(toy_retriever, llm, factory()), factory)
    without = app.build_app(RagPipeline(toy_retriever, None), None, factory)
    assert isinstance(with_llm, gr.Blocks) and isinstance(without, gr.Blocks)
    assert llm.calls == []


def test_ask_handler_retrieval_only(toy_retriever) -> None:
    text, rows, passages = app.ask_handler(RagPipeline(toy_retriever, None), "start the spindle clockwise")
    assert text.startswith("Top passages from the documentation:")
    assert len(rows) == 5 and [r[0] for r in rows] == [1, 2, 3, 4, 5]
    assert "**[1]" in passages
    assert app.ask_handler(RagPipeline(toy_retriever, None), "   ")[0] == "Type a question first."


def test_ask_handler_with_llm_numbers_citations(toy_retriever) -> None:
    pipeline = RagPipeline(toy_retriever, FakeLLM(["Dwell with G4 P- [1]."]))
    text, rows, _ = app.ask_handler(pipeline, "G4 dwell seconds")
    assert text == "Dwell with G4 P- [1]." and rows[0][:2] == [1, "G4 Dwell"]
    pipeline = RagPipeline(toy_retriever, FakeLLM(["NOT_IN_DOCS: no laser info."]))
    assert "not grounded" in app.ask_handler(pipeline, "laser power")[0]


def test_generate_handler(toy_retriever) -> None:
    generator = GcodeGenerator(toy_retriever, FakeLLM([VALID]), factory())
    program, status, issues, attempts = app.generate_handler(generator, "plunge 1 mm", "linuxcnc")
    assert program.startswith("G21") and status.startswith("**VALID**") and issues == []
    assert attempts.startswith("Attempts: 1")
    assert app.generate_handler(None, "anything", "linuxcnc")[1] == app.NO_LLM_GENERATE


def test_validate_handler() -> None:
    status, issues = app.validate_handler(factory, app.EXAMPLE_PROGRAM, "linuxcnc")
    assert status.startswith("**INVALID**")
    assert issues[0][:3] == ["E005", "error", 6]
    assert app.validate_handler(factory, "", "haas") == ("Paste a program first.", [])
