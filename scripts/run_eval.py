#!/usr/bin/env python
"""Run the retrieval evaluation (and the answer evaluation when an LLM is configured).

Writes results/results.json and results/results.md (SPEC 5.5).
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gcoderag.config import get_paths  # noqa: E402
from gcoderag.evaluate import CONFIGS, evaluate_config, load_questions, results_markdown  # noqa: E402
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


def main() -> int:
    """Evaluate every config and write the results files."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    paths = get_paths()
    questions = load_questions(paths.eval / "questions.jsonl")
    retriever = load_retriever()
    stats = json.loads((paths.corpus / "stats.json").read_text(encoding="utf-8"))

    configs, details = [], {}
    for cfg in CONFIGS:
        scores = evaluate_config(questions, retriever, cfg.mode)
        details[cfg.name] = scores.pop("per_question")
        configs.append({"name": cfg.name, "mode": cfg.mode, "description": cfg.description, **scores})
        print(f"{cfg.name:<15} recall@1={scores['recall_at_1']:.4f} recall@5={scores['recall_at_5']:.4f} "
              f"mrr={scores['mrr']:.4f}")
    best = max(configs, key=lambda c: (c["recall_at_5"], c["mrr"]))

    results = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "corpus": {"num_docs": stats["num_docs"], "num_chunks": stats["num_chunks"], "sources": stats["sources"]},
        "embedding_model": retriever.dense.embedding_model,
        "n_questions": len(questions),
        "configs": configs,
        "best_config": best["name"],
        "answer_eval": answer_eval_section(questions, retriever, paths.results),
        "per_question": details,
    }
    paths.results.mkdir(parents=True, exist_ok=True)
    (paths.results / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (paths.results / "results.md").write_text(results_markdown(results), encoding="utf-8")
    print(f"best: {best['name']}; answer_eval: {results['answer_eval'].get('status')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
