#!/usr/bin/env python
"""Generation benchmark: run the generator over eval/gen_requests.jsonl and write results/gen_results.json.

Needs an LLM (see scripts/check_env.py). Run it twice to compare few-shot examples:

    run_gen_eval.py --no-examples   # run label "without_examples"
    run_gen_eval.py                 # run label "with_examples" (2 similar library programs in the prompt)

Each run is stored under `runs[label]` in results/gen_results.json; the other run is kept.
Columns per request: first_attempt_ok (no validator errors on attempt 1), final_ok, attempts, warnings_final,
feed_below_z0 (the final program has a feed move below Z0, i.e. it actually cuts).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gcoderag.config import get_paths, load_settings  # noqa: E402
from gcoderag.examples import ExampleLibrary  # noqa: E402
from gcoderag.generator import GcodeGenerator  # noqa: E402
from gcoderag.genbench import load_requests, make_run, run_request, update_results_file  # noqa: E402
from gcoderag.index import load_retriever  # noqa: E402
from gcoderag.llm import get_llm_from_env, provider_and_model  # noqa: E402
from gcoderag.validator import GcodeValidator  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark once and update results/gen_results.json."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-examples", action="store_true", help="do not put library examples in the prompt")
    parser.add_argument("--limit", type=int, default=0, help="only run the first N requests (smoke test; not saved)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    load_settings()
    llm = get_llm_from_env()
    if llm is None:
        print("No LLM configured (see scripts/check_env.py): the generation benchmark needs one.", file=sys.stderr)
        return 2
    paths = get_paths()
    requests = load_requests(paths.eval / "gen_requests.jsonl")
    if args.limit:
        requests = requests[: args.limit]
    retriever = load_retriever()
    library = None if args.no_examples else ExampleLibrary.from_directory(retriever.embedder)
    generator = GcodeGenerator(retriever, llm, GcodeValidator(), library)
    label = "without_examples" if args.no_examples else "with_examples"

    rows = []
    for i, item in enumerate(requests, start=1):
        row = run_request(generator, item)
        rows.append(row)
        print(f"[{i:>2}/{len(requests)}] {row['id']} {row['dialect']:<8} first_ok={row['first_attempt_ok']!s:<5} "
              f"final_ok={row['final_ok']!s:<5} attempts={row['attempts']} warnings={row['warnings_final']} "
              f"cuts={row['feed_below_z0']}", flush=True)
    run = make_run(label, provider_and_model(llm)[1], rows)
    summary = {k: v for k, v in run.items() if k.endswith("_rate") or k.startswith("mean_") or k == "n"}
    print(f"{label}: {summary}")
    if args.limit:
        print("--limit given: results not saved")
        return 0
    update_results_file(paths.results / "gen_results.json", label, run)
    print(f"wrote {paths.results / 'gen_results.json'} (runs: {label})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
