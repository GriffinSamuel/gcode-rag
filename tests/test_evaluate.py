"""Tests for the retrieval metrics on hand-built rankings and the toy corpus."""
from __future__ import annotations

import pytest

from gcoderag.evaluate import (
    Question,
    evaluate_config,
    keywords_present,
    metrics_from_ranks,
    recall_at_k,
    reciprocal_rank,
    retrieval_by_split,
    select_split,
)


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


def test_keywords_match_whole_tokens_not_substrings() -> None:
    assert not keywords_present("Use M30 to end the program [1].", ["p"])
    assert keywords_present("Program G4 P2 to wait.", ["p"])
    assert not keywords_present("Use G10 L2 to set offsets.", ["g1"])
    assert not keywords_present("G1.5 is not a code.", ["g1"])
    assert keywords_present("Each block dwells; see block delete.", ["dwell", "block delete"])
    assert not keywords_present("Use modifiers.", ["if"])


def make_questions() -> list[Question]:
    return [Question(f"q{i:03d}", "code_lookup", f"question {i}", ["g"], ["k"], split)
            for i, split in enumerate(["dev", "test", "test", "dev", "test"], start=1)]


def test_metrics_from_ranks_matches_hand_math() -> None:
    m = metrics_from_ranks([1, 2, None, 6, 1])
    assert m == {"recall_at_1": 0.4, "recall_at_5": 0.6, "mrr": round((1 + 0.5 + 0 + 1 / 6 + 1) / 5, 4)}
    with pytest.raises(ValueError):
        metrics_from_ranks([])


def test_select_split() -> None:
    qs = make_questions()
    assert [q.id for q in select_split(qs, "dev")] == ["q001", "q004"]
    assert [q.id for q in select_split(qs, "test")] == ["q002", "q003", "q005"]
    assert select_split(qs, "all") == qs
    with pytest.raises(ValueError):
        select_split(qs, "train")


def test_retrieval_by_split_partitions_the_overall_numbers() -> None:
    qs = make_questions()
    rows = [{"id": q.id, "rank": r} for q, r in zip(qs, [1, None, 3, 2, 1])]
    by_split = retrieval_by_split(qs, rows)
    assert by_split["dev"] == {"n": 2, "recall_at_1": 0.5, "recall_at_5": 1.0, "mrr": 0.75}
    assert by_split["test"]["n"] == 3 and by_split["test"]["recall_at_1"] == pytest.approx(0.3333)
    overall = metrics_from_ranks([r["rank"] for r in rows])
    weighted = (2 * by_split["dev"]["recall_at_5"] + 3 * by_split["test"]["recall_at_5"]) / 5
    assert weighted == pytest.approx(overall["recall_at_5"], abs=1e-3)


def test_question_split_defaults_to_test_and_loads_from_json(tmp_path) -> None:
    from gcoderag.evaluate import load_questions

    assert Question("q001", "code_lookup", "x", ["g"], ["k"]).split == "test"
    path = tmp_path / "q.jsonl"
    path.write_text('{"id": "q001", "category": "syntax", "question": "x", "gold_doc_ids": ["g"], '
                    '"answer_keywords": ["k"], "split": "dev"}\n', encoding="utf-8")
    assert load_questions(path)[0].split == "dev"
