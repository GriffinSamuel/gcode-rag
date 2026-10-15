"""Tests for the RAG pipeline with FakeLLM on the toy corpus."""
from __future__ import annotations

import json

from gcoderag.evaluate import Question
from gcoderag.llm import FakeLLM
from gcoderag.rag import NOT_IN_DOCS, RagPipeline, cited_numbers, run_answer_eval, summarize_answers


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


def test_sources_line_sets_citations_and_is_stripped(toy_retriever) -> None:
    llm = FakeLLM(["Dwell with G4 P2.\nSOURCES: 1, 3, 9"])
    pipeline = RagPipeline(toy_retriever, llm)
    answer = pipeline.answer("G4 dwell seconds")
    assert answer.text == "Dwell with G4 P2." and answer.grounded
    assert [h.doc_id for h in answer.citations] == [answer.passages[0].doc_id, answer.passages[2].doc_id]
    assert "SOURCES: 1, 3" in llm.calls[0]["system"]


def test_without_sources_citations_come_from_codes_mentioned(toy_retriever) -> None:
    answer = RagPipeline(toy_retriever, FakeLLM(["Use G4 P2 to dwell."])).answer("pause two seconds")
    assert answer.grounded and "linuxcnc:gcode:g4" in [h.doc_id for p in [answer] for h in p.citations]
    assert all(h.doc_id == "linuxcnc:gcode:g4" or "G4" in h.codes for h in answer.citations)
    marker = RagPipeline(toy_retriever, FakeLLM(["Spindle on [2]."])).answer("spindle clockwise")
    assert marker.grounded and [h.doc_id for h in marker.citations] == [marker.passages[1].doc_id]


def test_without_sources_or_codes_top_passage_is_cited(toy_retriever) -> None:
    answer = RagPipeline(toy_retriever, FakeLLM(["It stops the motion at the end."])).answer("spindle clockwise")
    assert answer.grounded and answer.citations == answer.passages[:1]
    out_of_range = RagPipeline(toy_retriever, FakeLLM(["Nothing codelike here [9]."])).answer("rapid move")
    assert out_of_range.grounded and len(out_of_range.citations) == 1


def test_not_in_docs_ignores_sources_and_has_no_citations(toy_retriever) -> None:
    answer = RagPipeline(toy_retriever, FakeLLM([f"{NOT_IN_DOCS}: no laser info.\nSOURCES: 1"])).answer("laser")
    assert not answer.grounded and answer.citations == [] and "SOURCES" not in answer.text


def test_retrieval_only_mode(toy_retriever) -> None:
    answer = RagPipeline(toy_retriever, None).answer("spindle clockwise")
    assert answer.mode == "retrieval-only" and answer.grounded
    assert answer.text.startswith("Top passages from the documentation:")
    assert len(answer.citations) == 5
    assert "M3, M4, M5 Spindle Control" in answer.text and "https://" in answer.text


def test_cited_numbers_filters_range() -> None:
    assert cited_numbers("see [2], [1], [2] and [7]", 5) == [2, 1]


def test_run_answer_eval_scores_keywords_and_splits(toy_retriever, tmp_path) -> None:
    questions = [Question("q001", "code_lookup", "G4 dwell", ["linuxcnc:gcode:g4"], ["dwell", "p"], "dev"),
                 Question("q002", "code_lookup", "tool change", ["linuxcnc:mcode:m6"], ["m6"], "test"),
                 Question("q003", "code_lookup", "laser power", ["linuxcnc:gcode:g1"], ["g1"], "test")]
    llm = FakeLLM(["Dwell with G4 P2 [1].", "Use the T word [1].", "NOT_IN_DOCS: nothing about lasers."],
                  name="anthropic:claude-haiku-5-5")
    result = run_answer_eval(questions, toy_retriever, llm, tmp_path / "answers.jsonl", sleep_s=0)
    assert result == {"status": "done", "config": "hybrid_rrf", "provider": "anthropic", "model": "claude-haiku-5-5",
                      "n": 3, "keyword_accuracy": 0.3333, "citation_rate": 0.6667,
                      "by_split": {"dev": {"n": 1, "keyword_accuracy": 1.0, "citation_rate": 1.0},
                                   "test": {"n": 2, "keyword_accuracy": 0.0, "citation_rate": 0.5}}}
    rows = [json.loads(line) for line in (tmp_path / "answers.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["passed"] for r in rows] == [True, False, False]
    assert [r["split"] for r in rows] == ["dev", "test", "test"]


def test_summarize_answers_rates() -> None:
    rows = [{"passed": True, "citations": ["a"]}, {"passed": False, "citations": []}]
    assert summarize_answers(rows) == {"n": 2, "keyword_accuracy": 0.5, "citation_rate": 0.5}


def test_inline_sources_are_parsed_and_stripped() -> None:
    from gcoderag.rag import split_sources

    assert split_sources("The P word sets seconds. SOURCES: 1", 5) == ("The P word sets seconds.", [1])
    assert split_sources("Use G4.\nSOURCES: 2, 3 and 9", 5) == ("Use G4.", [2, 3])
    assert split_sources("Answer.\nSOURCES: [1], [2]", 5) == ("Answer.", [1, 2])
    assert split_sources("Check the sources: G4 is documented.", 5) == ("Check the sources: G4 is documented.", [])
