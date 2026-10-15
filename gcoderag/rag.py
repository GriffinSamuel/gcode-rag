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
from gcoderag.textnorm import extract_codes

logger = logging.getLogger(__name__)

RETRIEVE_K = 10
NOT_IN_DOCS = "NOT_IN_DOCS"
ANSWER_MAX_TOKENS = 400
EVAL_SLEEP_S = 0.5
SYSTEM_PROMPT = (
    "You are a CNC programming assistant. Answer ONLY from the numbered passages. Cite passages as [1], [2]. "
    "Keep answers under 120 words. If the passages do not contain the answer, reply exactly: "
    f"{NOT_IN_DOCS}: <one sentence saying what is missing>. "
    "Otherwise end your reply with one last line listing the passage numbers you used, for example: SOURCES: 1, 3"
)
RETRIEVAL_ONLY_HEADER = "Top passages from the documentation:"
_CITATION_RE = re.compile(r"\[(\d+)\]")
# "SOURCES: 1, 3" on its own line, or tacked onto the end of the last sentence ("... in seconds. SOURCES: 1")
_SOURCES_RE = re.compile(r"(?:^|[ \t]+)SOURCES?[ \t]*:[ \t]*((?:\[?\d+\]?(?:[ \t]*(?:,|;|and)[ \t]*)?)+)[ \t]*$",
                         re.IGNORECASE | re.MULTILINE)


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


def split_sources(text: str, n_passages: int) -> tuple[str, list[int]]:
    """Remove `SOURCES: 1, 3` lines from `text`; return the cleaned text and the valid passage numbers listed."""
    numbers: list[int] = []
    for match in _SOURCES_RE.finditer(text):
        numbers += [int(n) for n in re.findall(r"\d+", match.group(1))]
    cleaned = _SOURCES_RE.sub("", text).strip()
    return cleaned, list(dict.fromkeys(n for n in numbers if 1 <= n <= n_passages))


def choose_citations(text: str, passages: list[Hit], listed: list[int]) -> list[Hit]:
    """Deterministic citations: the SOURCES numbers, else [n] markers, else passages whose codes the answer
    mentions, else the top passage."""
    numbers = listed or cited_numbers(text, len(passages))
    if numbers:
        return [passages[n - 1] for n in numbers]
    mentioned = set(extract_codes(text))
    by_code = [p for p in passages if mentioned & set(p.codes)]
    return by_code or passages[:1]


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
        text, listed = split_sources(text, len(passages))
        if text.startswith(NOT_IN_DOCS):
            return Answer(text, [], False, "llm", passages)
        return Answer(text, choose_citations(text, passages, listed), True, "llm", passages)


def run_answer_eval(questions: list[Any], retriever: HybridRetriever, llm: LLM, answers_path: Path,
                    sleep_s: float = EVAL_SLEEP_S) -> dict[str, Any]:
    """SPEC 5.4: answer every question with hybrid_rrf, score keyword accuracy, save answers.jsonl."""
    from gcoderag.evaluate import keywords_present

    pipeline = RagPipeline(retriever, llm)
    rows: list[dict[str, Any]] = []
    for i, q in enumerate(questions):
        if i:
            time.sleep(sleep_s)
        try:
            answer = pipeline.answer(q.question)
        except Exception as exc:  # noqa: BLE001 - retries are exhausted inside the client
            logger.error("answer eval aborted at %s: %s", q.id, exc)
            return {"status": "skipped_no_llm_key", "reason": f"LLM error after retries: {type(exc).__name__}"}
        passed = keywords_present(answer.text, q.answer_keywords)
        rows.append({"id": q.id, "split": getattr(q, "split", "test"), "question": q.question, "answer": answer.text,
                     "citations": [h.doc_id for h in answer.citations], "passed": passed})
    answers_path.parent.mkdir(parents=True, exist_ok=True)
    answers_path.write_bytes("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8"))
    provider, model = provider_and_model(llm)
    return {"status": "done", "config": "hybrid_rrf", "provider": provider, "model": model, **summarize_answers(rows),
            "by_split": {split: summarize_answers([r for r in rows if r["split"] == split])
                         for split in ("dev", "test") if any(r["split"] == split for r in rows)}}


def summarize_answers(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """n, keyword accuracy and citation rate (share of answers with at least one citation) for answer rows."""
    n = len(rows)
    return {"n": n, "keyword_accuracy": round(sum(bool(r["passed"]) for r in rows) / n, 4),
            "citation_rate": round(sum(bool(r["citations"]) for r in rows) / n, 4)}
