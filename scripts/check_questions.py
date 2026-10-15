#!/usr/bin/env python
"""Validate eval/questions.jsonl against docs/EVAL.md and print a summary.

Checks: schema, unique ids, categories, gold ids exist in the corpus, keywords occur (as whole tokens, after
lowercasing and code normalization) in the gold text, minimum size and per-category counts, share of questions that
do not name a code, near-duplicate questions, and the dev/test split. Exits 1 if any check fails.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gcoderag.config import get_paths  # noqa: E402
from gcoderag.evaluate import keywords_present  # noqa: E402
from gcoderag.textnorm import CODE_RE  # noqa: E402

MIN_QUESTIONS = 160
MIN_PER_CORE_CATEGORY = 10
MIN_NO_CODE_FRACTION = 0.40
DEV_FRACTION_RANGE = (0.35, 0.45)
MIN_TWO_STEP = 5
DUPLICATE_JACCARD = 0.8
ALLOWED = {"code_lookup", "parameters", "modal_groups", "syntax", "canned_cycles", "safety_practice", "haas_specific"}
CORE = ["code_lookup", "parameters", "modal_groups", "syntax", "canned_cycles", "safety_practice"]
FIELDS = ("id", "category", "question", "gold_doc_ids", "answer_keywords", "split")
_ID_RE = re.compile(r"q\d{3}")


def words(text: str) -> set[str]:
    """Lowercase word set used for near-duplicate detection."""
    return set(re.findall(r"[a-z0-9.]+", text.lower()))


def jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity of two sets (0 when both are empty)."""
    return len(a & b) / len(a | b) if a | b else 0.0


def find_problems(questions: list[dict], docs: dict[str, str]) -> dict[str, list[str]]:
    """Return {check name: [problem, ...]} for the per-question checks."""
    problems: dict[str, list[str]] = {k: [] for k in ("schema", "ids", "categories", "gold ids", "keywords", "splits",
                                                        "duplicates")}
    seen: set[str] = set()
    for q in questions:
        qid = q.get("id", "?")
        if not all(k in q for k in FIELDS) or not q.get("question", "").strip():
            problems["schema"].append(f"{qid}: missing field")
            continue
        if not _ID_RE.fullmatch(qid) or qid in seen:
            problems["ids"].append(f"{qid}: bad or duplicate id")
        seen.add(qid)
        if q["category"] not in ALLOWED:
            problems["categories"].append(f"{qid}: {q['category']}")
        if not 1 <= len(q["gold_doc_ids"]) <= 3 or not 1 <= len(q["answer_keywords"]) <= 3:
            problems["schema"].append(f"{qid}: needs 1-3 gold ids and 1-3 keywords")
        missing = [g for g in q["gold_doc_ids"] if g not in docs]
        if missing:
            problems["gold ids"].append(f"{qid}: {missing}")
        gold_text = [docs[g] for g in q["gold_doc_ids"] if g in docs]
        absent = [k for k in q["answer_keywords"] if not any(keywords_present(t, [k]) for t in gold_text)]
        if absent:
            problems["keywords"].append(f"{qid}: {absent} not found in gold text")
        if q["split"] not in ("dev", "test"):
            problems["splits"].append(f"{qid}: split={q['split']!r}")
    word_sets = [(q.get("id", "?"), words(q.get("question", ""))) for q in questions]
    for i, (id_a, wa) in enumerate(word_sets):
        for id_b, wb in word_sets[i + 1:]:
            if jaccard(wa, wb) >= DUPLICATE_JACCARD:
                problems["duplicates"].append(f"{id_a} ~ {id_b}")
    return problems


def main() -> int:
    """Run all checks, print one line per check and a summary; return the exit code."""
    paths = get_paths()
    docs = {}
    with (paths.corpus / "docs.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                d = json.loads(line)
                docs[d["doc_id"]] = d["text"]
    with (paths.eval / "questions.jsonl").open(encoding="utf-8") as fh:
        questions = [json.loads(line) for line in fh if line.strip()]

    failures = 0

    def report(name: str, ok: bool, detail: str) -> None:
        nonlocal failures
        failures += not ok
        print(f"{'OK  ' if ok else 'FAIL'} {name}: {detail}")

    for name, items in find_problems(questions, docs).items():
        report(name, not items, "all good" if not items else f"{len(items)} problem(s), e.g. {items[:3]}")

    total = len(questions)
    report("total", total >= MIN_QUESTIONS, f"{total} questions (min {MIN_QUESTIONS})")
    by_cat = Counter(q["category"] for q in questions)
    low = {c: by_cat[c] for c in CORE if by_cat[c] < MIN_PER_CORE_CATEGORY}
    report("category counts", not low, ", ".join(f"{c}={by_cat[c]}" for c in CORE) + f" (min {MIN_PER_CORE_CATEGORY})")
    no_code = sum(1 for q in questions if not CODE_RE.search(q["question"]))
    report("no-code questions", no_code / total >= MIN_NO_CODE_FRACTION,
           f"{no_code}/{total} = {no_code / total:.0%} do not name a code (min {MIN_NO_CODE_FRACTION:.0%})")
    two_step = sum(1 for q in questions if len(q["gold_doc_ids"]) >= 2)
    report("multi-doc questions", two_step >= MIN_TWO_STEP, f"{two_step} questions with 2+ gold docs (min {MIN_TWO_STEP})")
    splits = Counter(q["split"] for q in questions)
    dev_frac = splits["dev"] / total
    report("split fraction", DEV_FRACTION_RANGE[0] <= dev_frac <= DEV_FRACTION_RANGE[1],
           f"dev={splits['dev']} ({dev_frac:.0%}), test={splits['test']} ({1 - dev_frac:.0%})")
    covered = {g for q in questions for g in q["gold_doc_ids"] if g in docs}
    print(f"\nSummary: {total} questions, {len(by_cat)} categories, {len(covered)}/{len(docs)} corpus docs used as "
          f"gold, splits {dict(splits)}")
    print("ALL CHECKS OK" if not failures else f"{failures} CHECK(S) FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
