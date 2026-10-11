"""Grounded question answering with numbered citations (SPEC 6.2) and the answer evaluation (SPEC 5.4)."""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from gcoderag.index import Hit, HybridRetriever
from gcoderag.llm import LLM, provider_and_model

logger = logging.getLogger(__name__)

RETRIEVE_K = 10
NOT_IN_DOCS = "NOT_IN_DOCS"
ANSWER_MAX_TOKENS = 400
EVAL_SLEEP_S = 0.5
SYSTEM_PROMPT = (
    "You are a CNC programming assistant. Answer ONLY from the numbered passages. Cite passages as [1], [2]. "
    "Keep answers under 120 words. If the passages do not contain the answer, reply exactly: "
    f"{NOT_IN_DOCS}: <one sentence saying what is missing>."
)
RETRIEVAL_ONLY_HEADER = "Top passages from the documentation:"
_CITATION_RE = re.compile(r"\[(\d+)\]")


@dataclass
class Answer:
    """An answer with the passages it cites."""

    text: str
    citations: list[Hit]
    grounded: bool
    mode: Literal["llm", "retrieval-only"]
    passages: list[Hit]


def best_chunk_per_doc(hits: list[Hit], limit: int) -> list[Hit]:
    """Keep the first (best-ranked) chunk of each distinct document, up to `limit` documents."""
    seen: dict[str, Hit] = {}
    for hit in hits:
        if hit.doc_id not in seen:
            seen[hit.doc_id] = hit
        if len(seen) == limit:
            break
    return list(seen.values())


def format_passages(passages: list[Hit]) -> str:
    """Render passages as numbered blocks with title, URL and text."""
    return "\n\n".join(f"[{i}] {p.title}\nURL: {p.url}\n{p.text}" for i, p in enumerate(passages, start=1))


def cited_numbers(text: str, n_passages: int) -> list[int]:
    """Distinct passage numbers cited as [n] with 1 <= n <= n_passages, in order of first appearance."""
    numbers = (int(m) for m in _CITATION_RE.findall(text))
    return list(dict.fromkeys(n for n in numbers if 1 <= n <= n_passages))


class RagPipeline:
    """Retrieve passages and (optionally) have an LLM answer from them with citations."""

    def __init__(self, retriever: HybridRetriever, llm: LLM | None) -> None:
        self.retriever = retriever
        self.llm = llm

    def answer(self, question: str, k: int = 5) -> Answer:
        """Answer `question` from the top `k` distinct documents."""
        passages = best_chunk_per_doc(self.retriever.retrieve(question, k=RETRIEVE_K, mode="hybrid"), k)
        if self.llm is None:
            lines = [RETRIEVAL_ONLY_HEADER, ""]
            for i, p in enumerate(passages, start=1):
                lines += [f"[{i}] **{p.title}** ({p.url})", p.text, ""]
            return Answer("\n".join(lines).strip(), list(passages), True, "retrieval-only", passages)
        user = f"{format_passages(passages)}\n\nQuestion: {question}"
        text = self.llm.complete(SYSTEM_PROMPT, user, max_tokens=ANSWER_MAX_TOKENS, temperature=0.0).strip()
        numbers = cited_numbers(text, len(passages))
        grounded = not text.startswith(NOT_IN_DOCS) and bool(numbers)
        return Answer(text, [passages[n - 1] for n in numbers], grounded, "llm", passages)


def run_answer_eval(questions: list[Any], retriever: HybridRetriever, llm: LLM, answers_path: Path,
                    sleep_s: float = EVAL_SLEEP_S) -> dict[str, Any]:
    """SPEC 5.4: answer every question with hybrid_rrf, score keyword accuracy, save answers.jsonl."""
    from gcoderag.evaluate import keywords_present

    pipeline = RagPipeline(retriever, llm)
    rows: list[dict[str, Any]] = []
    passes = 0
    for i, q in enumerate(questions):
        if i:
            time.sleep(sleep_s)
        try:
            answer = pipeline.answer(q.question)
        except Exception as exc:  # noqa: BLE001 - retries are exhausted inside the client
            logger.error("answer eval aborted at %s: %s", q.id, exc)
            return {"status": "skipped_no_llm_key", "reason": f"LLM error after retries: {type(exc).__name__}"}
        passed = keywords_present(answer.text, q.answer_keywords)
        passes += passed
        rows.append({"id": q.id, "question": q.question, "answer": answer.text,
                     "citations": [h.doc_id for h in answer.citations], "passed": passed})
    answers_path.parent.mkdir(parents=True, exist_ok=True)
    answers_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    provider, model = provider_and_model(llm)
    return {"status": "done", "config": "hybrid_rrf", "provider": provider, "model": model, "n": len(questions),
            "keyword_accuracy": round(passes / len(questions), 4)}
