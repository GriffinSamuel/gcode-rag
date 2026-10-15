"""Generation benchmark: Z0 detection, per-request rows, summary rates and the results file."""
from __future__ import annotations

import json

import pytest

from gcoderag.generator import GcodeGenerator
from gcoderag.genbench import (
    has_feed_move_below_zero,
    load_requests,
    make_run,
    run_request,
    summarize,
    update_results_file,
)
from gcoderag.llm import FakeLLM
from gcoderag.validator import GcodeValidator

CUTS = """G21 G90 G17
T1 M6
S1000 M3
G0 Z5
G1 Z-1 F100
G0 Z5
M5
M30
"""
VALID = "```gcode\n" + CUTS + "```"
NO_FEED = "```gcode\nG21 G90 G17\nT1 M6\nS1000 M3\nG0 Z5\nG1 Z-1\nG0 Z5\nM5\nM30\n```"


def test_feed_move_below_zero_detection() -> None:
    assert has_feed_move_below_zero(CUTS)
    assert not has_feed_move_below_zero("G21 G90\nG0 Z-5\nG1 Z2 F100\nG1 X10\nM30\n")       # rapid plunge is not a feed move
    assert not has_feed_move_below_zero("G21 G90\nG1 X10 F100\nG1 Z0\nM30\n")                # exactly 0 is not below
    assert has_feed_move_below_zero("G21 G90\nG0 Z2\nG91\nG1 Z-3 F50\nM30\n")                # incremental: 2 - 3 = -1
    assert not has_feed_move_below_zero("G21 G90\nG0 Z5\nG91\nG1 Z-3 F50\nM30\n")            # incremental: 5 - 3 = 2
    assert has_feed_move_below_zero("G21 G90\nG98 G81 X0 Y0 Z-8 R2 F100\nG80\nM30\n")        # canned cycle drills
    assert not has_feed_move_below_zero("G21 G90\nG10 L2 P1 Z-5\nG53 G0 Z-100\nM30\n")       # offsets / machine coords ignored
    assert not has_feed_move_below_zero("(only a comment)\n")


def make_generator(toy_retriever, responses):
    return GcodeGenerator(toy_retriever, FakeLLM(responses), GcodeValidator(use_corpus=False))


def test_run_request_records_first_attempt_failure_and_recovery(toy_retriever) -> None:
    row = run_request(make_generator(toy_retriever, [NO_FEED, VALID]), {"id": "g01", "dialect": "linuxcnc", "request": "cut"})
    assert row["first_attempt_ok"] is False and row["final_ok"] is True
    assert row["attempts"] == 2 and row["warnings_final"] == 0 and row["feed_below_z0"] is True
    assert row["errors_final"] == [] and row["program"].startswith("G21")


def test_run_request_first_attempt_ok(toy_retriever) -> None:
    row = run_request(make_generator(toy_retriever, [VALID]), {"id": "g02", "dialect": "linuxcnc", "request": "cut"})
    assert row["first_attempt_ok"] and row["attempts"] == 1


def test_run_request_records_llm_failure(toy_retriever) -> None:
    class Boom:
        name = "boom"

        def complete(self, *args, **kwargs):
            raise TimeoutError("model unreachable")

    gen = GcodeGenerator(toy_retriever, Boom(), GcodeValidator(use_corpus=False))
    row = run_request(gen, {"id": "g03", "dialect": "haas", "request": "cut"})
    assert row["final_ok"] is False and row["first_attempt_ok"] is False and "TimeoutError" in row["error"]
    assert row["warnings_final"] is None


def test_summarize_rates() -> None:
    rows = [
        {"first_attempt_ok": True, "final_ok": True, "attempts": 1, "warnings_final": 0, "feed_below_z0": True},
        {"first_attempt_ok": False, "final_ok": True, "attempts": 3, "warnings_final": 2, "feed_below_z0": True},
        {"first_attempt_ok": False, "final_ok": False, "attempts": 3, "warnings_final": 0, "feed_below_z0": False},
        {"first_attempt_ok": False, "final_ok": False, "attempts": 0, "warnings_final": None, "feed_below_z0": False},
    ]
    assert summarize(rows) == {"n": 4, "first_attempt_ok_rate": 0.25, "final_ok_rate": 0.5, "mean_attempts": 1.75,
                               "mean_warnings_final": 0.6667, "zero_warning_rate": 0.25, "feed_below_z0_rate": 0.5}
    with pytest.raises(ValueError):
        summarize([])


def test_update_results_file_keeps_the_other_run(tmp_path) -> None:
    row = {"id": "g01", "first_attempt_ok": True, "final_ok": True, "attempts": 1, "warnings_final": 0,
           "feed_below_z0": True}
    path = tmp_path / "out" / "gen_results.json"
    update_results_file(path, "with_examples", make_run("with_examples", "m", [row]))
    doc = update_results_file(path, "without_examples", make_run("without_examples", "m", [row]))
    assert list(doc["runs"]) == ["without_examples", "with_examples"]
    assert json.loads(path.read_text(encoding="utf-8"))["runs"]["with_examples"]["examples"] is True
    assert doc["runs"]["without_examples"]["examples"] is False
    with pytest.raises(ValueError):
        update_results_file(path, "other", {})


def test_gen_requests_file_is_well_formed() -> None:
    from gcoderag.config import get_paths

    rows = load_requests(get_paths().eval / "gen_requests.jsonl")
    assert len(rows) == 30 and len({r["id"] for r in rows}) == 30
    assert {r["dialect"] for r in rows} == {"linuxcnc", "haas"}
    assert sum("inch" in r["request"] for r in rows) >= 5
