# PROGRESS

Work strictly top to bottom. Tick a box only when its acceptance check passes. Commit after each item.
The Stop hook reads this file: unchecked boxes mean "not finished".

## Phase 0: Environment

- [x] 0.1 `git init -b main` (or `git init` + `git checkout -b main`); commit the scaffold as `0.1: scaffold`.
      Check: `git log --oneline | wc -l` >= 1 and `git branch --show-current` prints `main`.
- [x] 0.2 Create `.venv`, install `requirements.txt`, confirm `import sentence_transformers` works (or apply the
      fastembed fallback from CLAUDE.md section 5). Record Python version and embedding backend in `docs/DECISIONS.md`.
      Check: `.venv/Scripts/python.exe -c "import numpy, rank_bm25, gradio, pytest, requests, bs4"` exits 0.
- [x] 0.3 Run `scripts/check_env.py`; record the mode (`llm` with provider name, or `retrieval-only`) in `docs/DECISIONS.md`.
- [x] 0.4 Create package skeleton exactly as `docs/SPEC.md` section 2 (empty modules with docstrings), `pytest.ini`
      present, `tests/test_smoke.py` imports `gcoderag` and passes. Check: `pytest -q` shows `1 passed`.

## Phase 1: Corpus

- [x] 1.1 `gcoderag/ingest/linuxcnc.py` downloads the AsciiDoc sources (SPEC 3.1) into `data/raw/linuxcnc/` and parses
      them into documents (SPEC 3.3). Tests use the fixture snippet in `tests/fixtures/linuxcnc_snippet.adoc`.
      Check: `.venv/Scripts/python.exe -m gcoderag.ingest.linuxcnc` prints `docs: N` with N >= 120.
- [x] 1.2 `gcoderag/ingest/haas.py` (best effort, SPEC 3.2): robots check, download, parse, or write STATUS.json
      explaining the skip. Check: `data/raw/haas/STATUS.json` exists with `"status": "ok" | "skipped"`.
- [x] 1.3 `gcoderag/ingest/chunker.py` + `scripts/build_corpus.py` produce `data/corpus/docs.jsonl`,
      `data/corpus/chunks.jsonl`, `data/corpus/stats.json` (SPEC 3.4). Tests for parser and chunker pass.
      Check: chunks >= 150, distinct codes >= 60, every chunk has all required keys.

## Phase 2: Retrieval

- [x] 2.1 `gcoderag/embed.py`: `Embedder` protocol, `SentenceTransformerEmbedder`, `FastEmbedEmbedder`,
      `FakeEmbedder` (SPEC 4.1). Unit tests use `FakeEmbedder` only.
- [x] 2.2 `gcoderag/index.py`: `DenseIndex`, `BM25Index`, `HybridRetriever` with RRF and code normalization
      (SPEC 4.2-4.4); `scripts/build_index.py` writes `data/index/embeddings.npy` + `meta.json`.
      Check: tests pass; `meta.json.num_vectors` equals the chunk count.
- [x] 2.3 Smoke test: run `python -m gcoderag.cli search` for the 6 queries in SPEC 4.6 and confirm the expected
      doc appears in the top 5 for at least 5 of 6. Log the results table in `docs/DECISIONS.md`.

## Phase 3: Evaluation

- [x] 3.1 Write `eval/questions.jsonl` following `docs/EVAL.md`: >= 40 questions, >= 4 categories, every
      `gold_doc_ids` entry exists in `data/corpus/docs.jsonl`, 1-3 robust `answer_keywords` each.
- [x] 3.2 `gcoderag/evaluate.py` + `scripts/run_eval.py`: configs `dense_baseline`, `bm25_only`, `hybrid_rrf`
      (SPEC 5); writes `results/results.json` and `results/results.md`. Unit tests cover recall/MRR math on toy data.
- [x] 3.3 Best config Recall@5 >= 0.75. If not, improve per CLAUDE.md section 5 and re-run; log each iteration
      (before/after numbers) in `docs/DECISIONS.md`.

## Phase 4: Generation and validation

- [x] 4.1 `gcoderag/llm.py`: `LLM` protocol, `AnthropicLLM`, `OpenAICompatibleLLM`, `FakeLLM`, `get_llm_from_env()`
      (SPEC 6.1). Tests use `FakeLLM`.
- [x] 4.2 `gcoderag/rag.py`: grounded answers with `[n]` citations and the `NOT_IN_DOCS` refusal path (SPEC 6.2). Tests with `FakeLLM`.
- [x] 4.3 `gcoderag/validator.py`: parser + all rules E001-E010 and W101-W105 + CLI (SPEC 7). At least 15 unit
      tests, including one per rule.
- [x] 4.4 `gcoderag/generator.py`: generate -> validate -> retry (max 3 attempts) (SPEC 6.3). Test: `FakeLLM`
      returns an invalid program first, then a valid one; result reports `attempts == 2`.
- [x] 4.5 `examples/`: >= 3 `valid_*.nc` and >= 3 `invalid_*.nc`; `scripts/validate_examples.py` writes
      `results/validator_report.json` where every example matches its expected outcome.
- [x] 4.6 Answer evaluation (SPEC 5.4): if check_env reports `llm`, run it over all questions and fill
      `results.json.answer_eval` (`status: "done"`, keyword accuracy >= 0.60); otherwise set
      `status: "skipped_no_llm_key"`. Log the result.

## Phase 5: App and documentation

- [x] 5.1 `app.py`: Gradio app with tabs `Ask the manual`, `Generate G-code`, `Validate G-code` (SPEC 8). Works in
      retrieval-only mode when no key. Test: `tests/test_app.py` builds the Blocks object without launching.
- [x] 5.2 `README.md` per CLAUDE.md section 8 with real numbers from `results/results.json`.
- [x] 5.3 Deployment prepared per `docs/DEPLOY.md` (`deploy/README_HF.md` front matter, `requirements.txt`
      compatible). Deploy only if `HF_TOKEN` is set; otherwise leave exact instructions in README.
- [x] 5.4 `FINAL_REPORT.md` (>= 300 words) with headings `What was built`, `Results`, `Limitations`, `Demo`,
      written for the user to read before an interview: what to say, which numbers to quote, what to show.

## Phase 6: Verification

- [ ] 6.1 Run the `final-reviewer` subagent on the whole repo. Fix every finding it marks MUST-FIX. Log findings
      and fixes in `docs/DECISIONS.md`.
- [ ] 6.2 `.venv/Scripts/python.exe scripts/verify_done.py` prints `ALL CHECKS PASSED`; `git status --porcelain`
      is empty; every box above is checked.
