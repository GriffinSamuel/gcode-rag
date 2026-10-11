"""Tests for the RAG pipeline with FakeLLM on the toy corpus."""
from __future__ import annotations

import json

from gcoderag.evaluate import Question
from gcoderag.llm import FakeLLM
from gcoderag.rag import NOT_IN_DOCS, RagPipeline, cited_numbers, run_answer_eval


def test_grounded_answer_with_citations(toy_retriever) -> None:
    llm = FakeLLM(["G4 makes all axes dwell for P seconds [1]."])
    answer = RagPipeline(toy_retriever, llm).answer("G4 dwell seconds")
    assert answer.mode == "llm" and answer.grounded
    assert [h.doc_id for h in answer.citations] == ["linuxcnc:gcode:g4"]
    assert len(answer.passages) == 5 and len({h.doc_id for h in answer.passages}) == 5
    user_msg = llm.calls[0]["user"]
    assert user_msg.startswith("[1] G4 Dwell") and user_msg.rstrip().endswith("Question: G4 dwell seconds")
    assert "Answer ONLY from the numbered passages" in llm.calls[0]["system"]
    assert llm.calls[0]["temperature"] == 0.0


def test_not_in_docs_is_not_grounded(toy_retriever) -> None:
    llm = FakeLLM([f"{NOT_IN_DOCS}: the passages do not cover laser engraving [1]."])
    answer = RagPipeline(toy_retriever, llm).answer("laser engraving power")
    assert not answer.grounded


def test_answer_without_citation_is_not_grounded(toy_retriever) -> None:
    answer = RagPipeline(toy_retriever, FakeLLM(["Use G0 [9]."])).answer("rapid move")
    assert not answer.grounded and answer.citations == []


def test_retrieval_only_mode(toy_retriever) -> None:
    answer = RagPipeline(toy_retriever, None).answer("spindle clockwise")
    assert answer.mode == "retrieval-only" and answer.grounded
    assert answer.text.startswith("Top passages from the documentation:")
    assert len(answer.citations) == 5
    assert "M3, M4, M5 Spindle Control" in answer.text and "https://" in answer.text


def test_cited_numbers_filters_range() -> None:
    assert cited_numbers("see [2], [1], [2] and [7]", 5) == [2, 1]


def test_run_answer_eval_scores_keywords(toy_retriever, tmp_path) -> None:
    questions = [Question("q001", "code_lookup", "G4 dwell", ["linuxcnc:gcode:g4"], ["dwell", "p"]),
                 Question("q002", "code_lookup", "tool change", ["linuxcnc:mcode:m6"], ["m6"])]
    llm = FakeLLM(["Dwell with G4 P2 [1].", "Use the T word [1]."], name="anthropic:claude-haiku-5-5")
    result = run_answer_eval(questions, toy_retriever, llm, tmp_path / "answers.jsonl", sleep_s=0)
    assert result == {"status": "done", "config": "hybrid_rrf", "provider": "anthropic",
                      "model": "claude-haiku-5-5", "n": 2, "keyword_accuracy": 0.5}
    rows = [json.loads(line) for line in (tmp_path / "answers.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["passed"] for r in rows] == [True, False]
