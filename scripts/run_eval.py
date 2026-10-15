#!/usr/bin/env python
"""Run the retrieval evaluation (and the answer evaluation when an LLM is configured).

Writes results/results.json and results/results.md (SPEC 5.5) plus results/answers.jsonl.

Retrieval metrics are cheap, so they are always computed for every question (overall and per dev/test split). The
`--split` flag only restricts the answer evaluation (the slow, LLM-backed part) and what is printed:

    run_eval.py --split dev    tune on dev only; prints dev numbers only, answer_eval covers the dev questions
    run_eval.py --split test   answer evaluation on the held-out test questions
    run_eval.py --split all    the final run over every question (default); this is what results.json should hold
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gcoderag.config import get_paths  # noqa: E402
from gcoderag.evaluate import (  # noqa: E402
    CONFIGS,
    SPLITS,
    evaluate_config,
    load_questions,
    results_markdown,
    retrieval_by_split,
    select_split,
)
from gcoderag.index import load_retriever  # noqa: E402

logger = logging.getLogger("run_eval")


def answer_eval_section(questions: list, retriever: object, results_dir: Path) -> dict:
    """Run the SPEC 5.4 answer evaluation if an LLM is configured; else return the skipped marker."""
    try:
        from gcoderag.llm import get_llm_from_env
    except ImportError:
        return {"status": "skipped_no_llm_key"}
    llm = get_llm_from_env()
    if llm is None:
        return {"status": "skipped_no_llm_key"}
    from gcoderag.rag import run_answer_eval

    return run_answer_eval(questions, retriever, llm, results_dir / "answers.jsonl")


def build_by_split(configs: list[dict[str, Any]], per_config_split: dict[str, dict[str, Any]],
                   answer_eval: dict[str, Any]) -> dict[str, Any]:
    """results.json `by_split`: per split, retrieval metrics per config and (if it was run) the answer evaluation."""
    answers = answer_eval.get("by_split", {}) if answer_eval.get("status") == "done" else {}
    out: dict[str, Any] = {}
    for split in SPLITS:
        per_config = {c["name"]: per_config_split[c["name"]][split] for c in configs
                      if split in per_config_split[c["name"]]}
        if not per_config:
            continue
        n = next(iter(per_config.values()))["n"]
        out[split] = {"n": n, "configs": {name: {k: v for k, v in m.items() if k != "n"}
                                          for name, m in per_config.items()},
                      "answer_eval": answers.get(split)}
    return out


def print_scores(label: str, rows: dict[str, dict[str, Any]]) -> None:
    """Print one line per config for a set of questions."""
    for name, m in rows.items():
        print(f"{label:<5} {name:<15} recall@1={m['recall_at_1']:.4f} recall@5={m['recall_at_5']:.4f} mrr={m['mrr']:.4f}")


def main(argv: list[str] | None = None) -> int:
    """Evaluate every config and write the results files."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=("dev", "test", "all"), default="all",
                        help="questions used for the answer evaluation (default: all)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    paths = get_paths()
    questions = load_questions(paths.eval / "questions.jsonl")
    retriever = load_retriever()
    stats = json.loads((paths.corpus / "stats.json").read_text(encoding="utf-8"))

    configs, details, per_config_split = [], {}, {}
    for cfg in CONFIGS:
        scores = evaluate_config(questions, retriever, cfg.mode)
        details[cfg.name] = scores.pop("per_question")
        per_config_split[cfg.name] = retrieval_by_split(questions, details[cfg.name])
        configs.append({"name": cfg.name, "mode": cfg.mode, "description": cfg.description, **scores})
    best = max(configs, key=lambda c: (c["recall_at_5"], c["mrr"]))

    shown = SPLITS if args.split == "all" else (args.split,)
    if args.split == "all":
        print_scores("all", {c["name"]: c for c in configs})
    for split in shown:
        print_scores(split, {name: per_config_split[name][split] for name in per_config_split})

    answer_eval = answer_eval_section(select_split(questions, args.split), retriever, paths.results)
    results = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "corpus": {"num_docs": stats["num_docs"], "num_chunks": stats["num_chunks"], "sources": stats["sources"]},
        "embedding_model": retriever.dense.embedding_model,
        "n_questions": len(questions),
        "split_evaluated": args.split,
        "configs": configs,
        "best_config": best["name"],
        "answer_eval": answer_eval,
        "by_split": build_by_split(configs, per_config_split, answer_eval),
        "per_question": details,
    }
    if answer_eval.get("status") == "done":
        results["citation_rate"] = answer_eval["citation_rate"]
    paths.results.mkdir(parents=True, exist_ok=True)
    (paths.results / "results.json").write_bytes(json.dumps(results, indent=2).encode("utf-8"))
    (paths.results / "results.md").write_bytes(results_markdown(results).encode("utf-8"))
    if answer_eval.get("status") == "done":
        for split, row in answer_eval["by_split"].items():
            print(f"{split:<5} answers: keyword_accuracy={row['keyword_accuracy']:.4f} "
                  f"citation_rate={row['citation_rate']:.4f} (n={row['n']})")
        if args.split == "all":
            print(f"all   answers: keyword_accuracy={answer_eval['keyword_accuracy']:.4f} "
                  f"citation_rate={answer_eval['citation_rate']:.4f} (n={answer_eval['n']})")
    else:
        print(f"answer_eval: {answer_eval.get('status')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
