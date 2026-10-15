"""Retrieval evaluation: Recall@k and MRR across retrieval configs (SPEC 5); keyword-based answer evaluation."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from gcoderag.ingest import read_jsonl
from gcoderag.textnorm import normalize_codes

logger = logging.getLogger(__name__)

RETRIEVAL_K = 10
DECIMALS = 4


@dataclass(frozen=True)
class Question:
    """One evaluation question (docs/EVAL.md schema)."""

    id: str
    category: str
    question: str
    gold_doc_ids: list[str]
    answer_keywords: list[str]
    split: str = "test"   # "dev" (used for tuning) or "test" (held out); see docs/EVAL.md


SPLITS = ("dev", "test")


@dataclass(frozen=True)
class RetrievalConfig:
    """A named retrieval configuration compared in the evaluation."""

    name: str
    mode: str
    description: str


CONFIGS: tuple[RetrievalConfig, ...] = (
    RetrievalConfig("dense_baseline", "dense", "bge-small cosine only"),
    RetrievalConfig("bm25_only", "bm25", "BM25 over normalized tokens"),
    RetrievalConfig("hybrid_rrf", "hybrid", "dense + BM25 fused with RRF k=60"),
)


class DocRetriever(Protocol):
    """The part of HybridRetriever the evaluation needs."""

    def retrieve_docs(self, query: str, k: int = 10, mode: Any = "hybrid") -> list[str]:
        """Distinct doc ids in rank order."""
        ...


def load_questions(path: Path) -> list[Question]:
    """Load `eval/questions.jsonl`."""
    return [Question(**row) for row in read_jsonl(path)]


def recall_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    """1.0 if any gold id is among the first `k` ranked ids, else 0.0."""
    return 1.0 if any(doc_id in gold for doc_id in ranked[:k]) else 0.0


def reciprocal_rank(ranked: list[str], gold: set[str], max_rank: int = 10) -> float:
    """1 / rank of the first gold id within `max_rank`, else 0.0 (ranks start at 1)."""
    for rank, doc_id in enumerate(ranked[:max_rank], start=1):
        if doc_id in gold:
            return 1.0 / rank
    return 0.0


def evaluate_config(questions: list[Question], retriever: DocRetriever, mode: str) -> dict[str, Any]:
    """Recall@1, Recall@5 and MRR for one retrieval mode, plus per-question first-gold ranks."""
    if not questions:
        raise ValueError("no questions to evaluate")
    per_question: list[dict[str, Any]] = []
    for q in questions:
        ranked = retriever.retrieve_docs(q.question, k=RETRIEVAL_K, mode=mode)
        rr = reciprocal_rank(ranked, set(q.gold_doc_ids), max_rank=RETRIEVAL_K)
        per_question.append({"id": q.id, "rank": round(1 / rr) if rr else None, "top": ranked[:3]})
    return {**metrics_from_ranks([row["rank"] for row in per_question]), "per_question": per_question}


def metrics_from_ranks(ranks: list[int | None]) -> dict[str, float]:
    """Recall@1, Recall@5 and MRR from the rank of the first gold doc per question (None = not in the top 10)."""
    if not ranks:
        raise ValueError("no questions to evaluate")
    n = len(ranks)
    return {
        "recall_at_1": round(sum(1.0 for r in ranks if r == 1) / n, DECIMALS),
        "recall_at_5": round(sum(1.0 for r in ranks if r is not None and r <= 5) / n, DECIMALS),
        "mrr": round(sum(1.0 / r for r in ranks if r is not None) / n, DECIMALS),
    }


def select_split(questions: list[Question], split: str) -> list[Question]:
    """Questions of one split ("dev" or "test"); "all" returns every question."""
    if split == "all":
        return list(questions)
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS + ('all',)}, got {split!r}")
    return [q for q in questions if q.split == split]


def retrieval_by_split(questions: list[Question], per_question: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Retrieval metrics per split, from the per-question ranks of one config (same math as the overall numbers)."""
    rank_of = {row["id"]: row["rank"] for row in per_question}
    out: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        ranks = [rank_of[q.id] for q in questions if q.split == split]
        if ranks:
            out[split] = {"n": len(ranks), **metrics_from_ranks(ranks)}
    return out


def keyword_pattern(keyword: str) -> re.Pattern[str]:
    """Token-boundary pattern for a normalized, lowercased keyword.

    Always anchored at the start of a token. A keyword ending in a digit (a code such as `g1`) must not continue with
    more digits (`g10`, `g1.5`); a 1-2 character keyword (`p`) must not continue with letters (`program`); longer words
    may take a suffix (`dwell` matches `dwells`).
    """
    if keyword[-1].isdigit():
        tail = r"(?!\d|\.\d)"
    elif len(keyword) <= 2:
        tail = r"(?![a-z])"
    else:
        tail = ""
    return re.compile(rf"(?<![a-z0-9]){re.escape(keyword)}{tail}")


def keywords_present(answer: str, keywords: list[str]) -> bool:
    """True if every keyword occurs as a token in the answer after code normalization and lowercasing."""
    haystack = normalize_codes(answer).lower()
    return all(keyword_pattern(normalize_codes(kw).lower()).search(haystack) for kw in keywords)


def results_markdown(results: dict[str, Any]) -> str:
    """Render results.json as the markdown table pasted into the README."""
    lines = [
        "| config | recall@1 | recall@5 | MRR |",
        "| --- | --- | --- | --- |",
        *(f"| {c['name']} | {c['recall_at_1']:.4f} | {c['recall_at_5']:.4f} | {c['mrr']:.4f} |" for c in results["configs"]),
        "",
        f"- Questions: {results['n_questions']}; embedding model: {results['embedding_model']}; "
        f"corpus: {results['corpus']['num_docs']} docs / {results['corpus']['num_chunks']} chunks "
        f"(sources: {results['corpus']['sources']})",
        f"- Best config by recall@5: {results['best_config']}",
    ]
    by_split = results.get("by_split") or {}
    if by_split:
        lines += ["", "| split | n | config | recall@1 | recall@5 | MRR |", "| --- | --- | --- | --- | --- | --- |"]
        for split, block in by_split.items():
            lines += [f"| {split} | {block['n']} | {name} | {m['recall_at_1']:.4f} | {m['recall_at_5']:.4f} | "
                      f"{m['mrr']:.4f} |" for name, m in block["configs"].items()]
        lines.append("")
    answer_eval = results.get("answer_eval", {})
    if answer_eval.get("status") == "done":
        lines.append(f"- Answer evaluation ({answer_eval['provider']} {answer_eval['model']}, {answer_eval['config']}): "
                     f"keyword accuracy {answer_eval['keyword_accuracy']:.4f} over {answer_eval['n']} questions"
                     + (f", citation rate {answer_eval['citation_rate']:.4f}" if "citation_rate" in answer_eval else ""))
        for split, row in (answer_eval.get("by_split") or {}).items():
            lines.append(f"  - {split}: keyword accuracy {row['keyword_accuracy']:.4f}, citation rate "
                         f"{row['citation_rate']:.4f} (n={row['n']})")
    else:
        lines.append("- Answer evaluation skipped: no LLM key configured")
    return "\n".join(lines) + "\n"
