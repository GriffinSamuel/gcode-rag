# gcode-rag: CNC G-Code Assistant (RAG + validated generation)

Portfolio project for NVIDIA Ignite / Microsoft Explore applications. Built in ONE unattended
Claude Code session. You (the agent) own the whole build from empty folder to finished repo.

## 1. Mission (read first)

Build a Retrieval-Augmented Generation system over open CNC G-code / M-code documentation that:
1. Answers machinist questions with citations to the exact documentation section.
2. Generates G-code programs from natural-language requests, then validates them with a
   deterministic rule-based validator (modal conflicts, missing feed, envelope, etc.) and retries on errors.
3. Is measured: a 40+ question evaluation set, Recall@5 and MRR for three retrieval configs
   (dense, BM25, hybrid RRF), and an answer-accuracy score when an LLM key is available.
4. Ships as a Gradio web app with a README that reports real numbers.

Full technical specification: `docs/SPEC.md`. Evaluation design: `docs/EVAL.md`. Deployment: `docs/DEPLOY.md`.
Work checklist: `PROGRESS.md`. Decision log: `docs/DECISIONS.md`. Definition of done: `scripts/verify_done.py`.

## 2. Hard constraints

- Machine: Windows 11, Claude Code runs commands through Git Bash. Use forward slashes. Never rely on
  `source .venv/bin/activate`; always call `.venv/Scripts/python.exe` explicitly.
- Everything must work offline-after-setup except: downloading docs, pip installs, embedding-model
  download, LLM API calls, optional Hugging Face deploy. No GPU. No paid services beyond the user's own LLM key.
- Python 3.10+ only. No notebooks. Stdlib + the packages in `requirements.txt` (add more only if needed, pinned loosely).
- Never read, print, edit, or commit `.env`. Check configuration only via `scripts/check_env.py`.
- Never create a git remote, push, force-push, reset --hard, or clean. The user publishes the repo later.
- Haas content is copyrighted: the scraper and the processed corpus stay local. All of `data/` is gitignored;
  the corpus and index are rebuilt by `scripts/build_corpus.py` and `scripts/build_index.py` (document this in the README).

## 3. Unattended operating protocol

You are running with NO human available. These rules override your defaults:

1. NEVER stop to ask the user a question. If something is ambiguous, pick the option that best matches
   `docs/SPEC.md`, write one line in `docs/DECISIONS.md` (format: `- YYYY-MM-DD HH:MM | <topic> | <decision> | <why>`), and continue.
2. Work through `PROGRESS.md` strictly in order. Tick a box (`- [ ]` -> `- [x]`) the moment the item's
   acceptance check passes. Commit after every completed item: `git add -A && git commit -m "<phase>.<item>: <summary>"`.
3. A Stop hook runs `scripts/verify_done.py` every time you try to end your turn and will refuse to let
   you stop until it prints `ALL CHECKS PASSED`. A `/goal` is also active. Do not fight them; keep working.
4. Before you believe you are finished: run `.venv/Scripts/python.exe scripts/verify_done.py` yourself,
   read every FAIL line, fix, repeat until it passes. Then run the `final-reviewer` subagent (Phase 6).
5. After any context compaction: re-read this file, `PROGRESS.md`, and the last 20 lines of
   `docs/DECISIONS.md` before touching code.
6. Do not refactor finished phases unless a test or verify_done fails. Do not add features not in SPEC.
7. Keep a steady cadence: small edits, run the relevant tests, commit. Never leave the tree dirty at a stop.

## 4. Escalation protocol (when you hit a wall)

Trigger: the SAME step fails twice (same error twice, a test still red after two fixes, a dependency that
will not install after two attempts, a metric below threshold after two improvement attempts), OR the Stop
hook tells you there has been no progress.

Action: immediately delegate to the `fable-escalation` subagent (strongest model, max effort). Your brief MUST include:
- The PROGRESS item id and the SPEC section involved
- The exact command(s) run and the complete error text (verbatim, not summarized)
- What you already tried (each attempt) and the current hypothesis
- The files involved (paths) and the acceptance check that must pass

Then apply/keep its changes, re-run the check, log the outcome in `docs/DECISIONS.md`, and continue.
Do not escalate the same blocker more than twice; after the second escalation use the fallback below.

## 5. Nothing is impossible: the fallback ladder

| If this fails | Do this instead (and log it) |
| --- | --- |
| Haas pages fail to download, are JS-rendered (no `G01` text in raw HTML), or robots.txt disallows | Skip Haas entirely. Write `data/raw/haas/STATUS.json` with the reason. Corpus = LinuxCNC only. All thresholds in verify_done are satisfiable with LinuxCNC alone. |
| A LinuxCNC `.adoc` file returns 404 | Skip that file; the 3 required ones are `g-code.adoc`, `m-code.adoc`, `overview.adoc`. |
| `sentence-transformers`/`torch` will not install or import | `pip install fastembed` and use `FastEmbedEmbedder` with the same model name (`BAAI/bge-small-en-v1.5`). |
| Embedding model download fails | Retry once; then use `sentence-transformers/all-MiniLM-L6-v2`; record the model in `data/index/meta.json`. |
| No LLM key configured (check_env says `retrieval-only`) | Build everything with `FakeLLM` in tests; app runs in retrieval-only mode (shows passages + citations); `results.json.answer_eval.status = "skipped_no_llm_key"`. |
| LLM API errors (rate limit, auth) | Retry with exponential backoff 3x; then fall back to retrieval-only mode as above and log it. |
| Hybrid Recall@5 < 0.75 after two attempts | Check: code normalization (G01->G1) applied to both chunks and queries; chunk size 200-400 words; k=10 before dedupe; question gold ids correct. Escalate if still low. |
| `HF_TOKEN` missing | Do not deploy. Finish `docs/DEPLOY.md` steps as copy-paste instructions for the user. |
| `gradio` will not install | Use `streamlit` with the same three tabs; update README and DEPLOY. |

## 6. Environment and commands (Git Bash on Windows)

```bash
python -m venv .venv                      # if "python" is missing try: py -3 -m venv .venv
.venv/Scripts/python.exe -m pip install --upgrade pip
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe scripts/check_env.py          # LLM/HF configuration status, no secrets printed
.venv/Scripts/python.exe -m pytest -q                  # fast tests (slow/network tests skipped by default)
RUN_SLOW=1 .venv/Scripts/python.exe -m pytest -q       # includes real-embedding integration tests
.venv/Scripts/python.exe scripts/build_corpus.py       # -> data/corpus/docs.jsonl, chunks.jsonl
.venv/Scripts/python.exe scripts/build_index.py        # -> data/index/
.venv/Scripts/python.exe scripts/run_eval.py           # -> results/results.json, results/results.md
.venv/Scripts/python.exe scripts/validate_examples.py  # -> results/validator_report.json
.venv/Scripts/python.exe -m gcoderag.cli search "what does G4 do"
.venv/Scripts/python.exe -m gcoderag.validator examples/valid_square_pocket.nc --dialect linuxcnc
.venv/Scripts/python.exe app.py                        # Gradio app on http://127.0.0.1:7860
.venv/Scripts/python.exe scripts/verify_done.py        # definition of done
```

## 7. Code standards

- Package `gcoderag/`, type hints everywhere, `from __future__ import annotations`, dataclasses for records,
  docstring on every public function, no global mutable state, logging via `logging` (no bare prints in library code).
- Pure functions for parsing/validation so they are unit-testable without network or models.
- Tests in `tests/` with pytest. Network/model tests use `@pytest.mark.slow` + `skipif(not os.getenv("RUN_SLOW"))`.
  Everything else must run offline in under 60 seconds using `FakeEmbedder` and `FakeLLM`.
- Deterministic outputs: `temperature=0`, fixed RRF `k=60`, fixed random seeds.
- Every file read/write uses `encoding="utf-8"`.
- Commit messages: `0.2: create venv and install requirements` style (phase.item: summary).

## 8. README requirements (checked by verify_done)

Sections (exact heading words, any level): `Architecture`, `How to run`, `Evaluation`, `Results`, `Safety`,
`Decisions`. The Results section must contain a markdown table with Recall@5 and MRR for all three configs copied
from `results/results.json`. The Safety section must contain this sentence verbatim:
"Generated G-code is for learning and must be checked in a simulator (such as NC Viewer) and reviewed by a
qualified machinist before it is run on a real machine."

## 9. Definition of done

`scripts/verify_done.py` prints `ALL CHECKS PASSED`, every box in `PROGRESS.md` is checked, `git status --porcelain`
is empty, and `FINAL_REPORT.md` exists. Nothing else counts as done.
