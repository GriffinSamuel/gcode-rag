"""Tests for the generate -> validate -> retry loop with a scripted FakeLLM."""
from __future__ import annotations

import pytest

from gcoderag.generator import FIX_PROMPT, GcodeGenerator, extract_program
from gcoderag.llm import FakeLLM
from gcoderag.validator import GcodeValidator

INVALID = """Here is the program:
```gcode
G21 G90 G17
T1 M6
S10000 M3
G0 Z5
G1 Z-1
G1 X20
M5
M30
```"""
VALID = """```gcode
G21 G90 G17
T1 M6
S10000 M3
G0 Z5
G1 Z-1 F200
G1 X20
G0 Z5
M5
M30
```"""


def make_generator(toy_retriever, responses: list[str]) -> tuple[GcodeGenerator, FakeLLM]:
    llm = FakeLLM(responses)
    return GcodeGenerator(toy_retriever, llm, GcodeValidator(use_corpus=False)), llm


def test_invalid_then_valid_takes_two_attempts(toy_retriever) -> None:
    gen, llm = make_generator(toy_retriever, [INVALID, VALID])
    result = gen.generate("face a 20 mm line 1 mm deep")
    assert result.attempts == 2 and result.report.ok
    assert result.program.startswith("G21 G90 G17") and "```" not in result.program
    assert [h["ok"] for h in result.history] == [False, True]
    assert any("E005" in e for e in result.history[0]["errors"])
    retry_msg = llm.calls[1]["user"]
    assert FIX_PROMPT in retry_msg and "E005" in retry_msg and "G1 Z-1" in retry_msg
    assert "Request: face a 20 mm line 1 mm deep" in llm.calls[0]["user"]
    assert "Reference passages:" in llm.calls[0]["user"]


def test_valid_first_time(toy_retriever) -> None:
    gen, llm = make_generator(toy_retriever, [VALID])
    result = gen.generate("cut a line")
    assert result.attempts == 1 and result.report.ok and len(llm.calls) == 1
    assert "LinuxCNC" in llm.calls[0]["system"] and "M30" in llm.calls[0]["system"]


def test_gives_up_after_max_attempts(toy_retriever) -> None:
    gen, llm = make_generator(toy_retriever, [INVALID])
    result = gen.generate("cut a line", max_attempts=3)
    assert result.attempts == 3 and not result.report.ok and len(llm.calls) == 3


def test_haas_dialect_prompt_and_validator(toy_retriever) -> None:
    haas = "```gcode\n%\nO00001\n" + VALID.split("\n", 1)[1].replace("```", "%\n```")
    gen, llm = make_generator(toy_retriever, [haas])
    result = gen.generate("cut a line", dialect="haas")
    assert result.dialect == "haas" and result.report.dialect == "haas" and result.report.ok
    assert result.report.warnings == []
    assert "O00001" in llm.calls[0]["system"]


def test_no_llm_raises(toy_retriever) -> None:
    gen = GcodeGenerator(toy_retriever, None, GcodeValidator(use_corpus=False))
    with pytest.raises(RuntimeError, match="LLM not configured"):
        gen.generate("anything")


def test_extract_program_variants() -> None:
    assert extract_program("```\nG0 X1\n```") == "G0 X1"
    assert extract_program("```text\nnotes\n```\n```nc\nG1 X2\n```") == "G1 X2"
    assert extract_program("G0 X3") == "G0 X3"


WARN_ONLY = """```gcode
G90 G17
T1 M6
S10000 M3
G0 Z5
G1 Z-1 F200
G1 X20
G0 Z5
M5
M30
```"""


def test_warning_feedback_second_request_cleans_program(toy_retriever) -> None:
    gen, llm = make_generator(toy_retriever, [WARN_ONLY, VALID])
    result = gen.generate("cut a line")
    assert result.attempts == 2 and result.report.ok and result.report.warnings == []
    assert [(h["n_errors"], h["n_warnings"]) for h in result.history] == [(0, 1), (0, 0)]
    assert result.history[0]["warnings"][0].startswith("W105")
    retry = llm.calls[1]["user"]
    assert "The program is valid but has these warnings; fix them if possible and return the complete program:" in retry
    assert "W105" in retry and "Your previous program" in retry


def test_warning_request_is_sent_only_once(toy_retriever) -> None:
    gen, llm = make_generator(toy_retriever, [WARN_ONLY])
    result = gen.generate("cut a line", max_attempts=3)
    assert result.attempts == 2 and len(llm.calls) == 2
    assert result.report.ok and len(result.report.warnings) == 1


def test_no_warning_request_when_no_attempts_remain(toy_retriever) -> None:
    gen, llm = make_generator(toy_retriever, [WARN_ONLY, VALID])
    result = gen.generate("cut a line", max_attempts=1)
    assert result.attempts == 1 and len(llm.calls) == 1 and len(result.report.warnings) == 1


def test_worse_retry_does_not_replace_better_program(toy_retriever) -> None:
    gen, _ = make_generator(toy_retriever, [WARN_ONLY, INVALID])
    result = gen.generate("cut a line")
    assert result.attempts == 3
    assert result.report.ok and len(result.report.warnings) == 1 and result.program.startswith("G90 G17")
    assert [h["n_errors"] > 0 for h in result.history] == [False, True, True]
