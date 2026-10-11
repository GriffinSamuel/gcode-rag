"""Tests for the retrieval metrics on hand-built rankings and the toy corpus."""
from __future__ import annotations

import pytest

from gcoderag.evaluate import Question, evaluate_config, keywords_present, recall_at_k, reciprocal_rank


def test_recall_at_k() -> None:
    ranked = ["a", "b", "c", "d", "e", "f"]
    assert recall_at_k(ranked, {"a"}, 1) == 1.0
    assert recall_at_k(ranked, {"c"}, 1) == 0.0
    assert recall_at_k(ranked, {"e", "z"}, 5) == 1.0
    assert recall_at_k(ranked, {"f"}, 5) == 0.0
    assert recall_at_k([], {"a"}, 5) == 0.0


def test_reciprocal_rank() -> None:
    ranked = ["a", "b", "c"]
    assert reciprocal_rank(ranked, {"a"}) == 1.0
    assert reciprocal_rank(ranked, {"c", "b"}) == pytest.approx(0.5)
    assert reciprocal_rank(ranked, {"z"}) == 0.0
    assert reciprocal_rank(ranked, {"c"}, max_rank=2) == 0.0


def test_evaluate_config_on_toy_corpus(toy_retriever) -> None:
    questions = [
        Question("q001", "code_lookup", "G4 dwell seconds", ["linuxcnc:gcode:g4"], ["dwell"]),
        Question("q002", "code_lookup", "tool change T word", ["linuxcnc:mcode:m6"], ["m6"]),
        Question("q003", "code_lookup", "zzz unrelated", ["linuxcnc:none"], ["x"]),
    ]
    result = evaluate_config(questions, toy_retriever, "bm25")
    assert result["recall_at_1"] == pytest.approx(0.6667)
    assert result["recall_at_5"] == pytest.approx(0.6667)
    assert result["mrr"] == pytest.approx(0.6667)
    assert [row["rank"] for row in result["per_question"]] == [1, 1, None]


def test_keywords_present_normalizes_codes() -> None:
    assert keywords_present("Use G01 with an F word.", ["g1", "f"])
    assert not keywords_present("Use G0.", ["g1"])
