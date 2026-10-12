# gcode-rag: a CNC G-code assistant with cited answers and validated generation

Ask machining questions and get answers grounded in the LinuxCNC G-code documentation, with a citation to the
exact manual section. Describe a part and get a G-code program that a deterministic validator has checked
(modal conflicts, missing feed, envelope limits, and more). If the validator finds errors, the program is sent back
to the model with the error list, up to 3 attempts. The retrieval is measured on a 56-question evaluation set
that compares three retrieval configurations.

- **Retrieval:** dense embeddings (`BAAI/bge-small-en-v1.5`) + BM25, fused with Reciprocal Rank Fusion (k = 60)
- **Generation:** pluggable LLM (Anthropic, or any OpenAI-compatible endpoint such as Groq, Gemini or Ollama)
- **Validation:** rule-based Python with 16 rules, no LLM in the loop, 31 unit tests
- **UI:** Gradio app with three tabs: *Ask the manual*, *Generate G-code*, *Validate G-code*

Headline numbers (from [results/results.json](results/results.json)): hybrid retrieval puts a correct manual
section in the top 5 for **94.6 %** of questions (Recall@5 0.9464) and ranks it first for **75 %** (Recall@1
0.75, MRR 0.8363).
Answer accuracy has not been measured yet because this build ran without an LLM key (details under Results).

## Architecture

```
               LinuxCNC AsciiDoc sources (g-code, m-code, overview, o-code, ... 9 files)
                                      |
             gcoderag/ingest   parse "== heading" sections -> 150 docs -> 276 chunks (median 191 words)
                                      |
                 +--------------------+---------------------+
                 |                                          |
       dense index (bge-small, 384-d,           BM25 index (rank_bm25 over
       cosine on normalized vectors)            code-normalized tokens: G01 -> G1)
                 |                                          |
                 +------------ RRF fusion, k = 60 ----------+
                                      |
                        top 10 chunks -> best chunk of the first 5 distinct docs
                                      |
          +---------------------------+------------------------------+
          |                                                          |
   rag.py: answer ONLY from passages [1]..[5],             generator.py: system prompt with dialect rules
   cite as [n], or reply NOT_IN_DOCS                       + 6 passages -> LLM -> ```gcode block
   (no key: show the passages + links)                                   |
                                                           validator.py (E001-E010, W101-W106)
                                                             ok? -> return    errors? -> send the errors back, retry (max 3)
```

| Module | Role |
| --- | --- |
| `gcoderag/ingest/linuxcnc.py` | Downloads the LinuxCNC `.adoc` sources from GitHub and splits them into one document per level-2 section, keeping the section anchor for citation URLs |
| `gcoderag/ingest/haas.py` | Best-effort Haas scraper (robots check, 1 request/s). Haas returned HTTP 403, so the corpus is LinuxCNC only (see Decisions) |
| `gcoderag/ingest/chunker.py` | Paragraph packing to a 250-word target (max 400, 40-word overlap); median chunk 191 words |
| `gcoderag/textnorm.py` | `normalize_codes` (`G01`/`g 01` -> `G1`, `G38.2` unchanged) applied to chunks *and* queries |
| `gcoderag/index.py` | `DenseIndex`, `BM25Index`, `HybridRetriever` (RRF, ties broken by dense score) |
| `gcoderag/rag.py` | Grounded answers with `[n]` citations and a `NOT_IN_DOCS` refusal |
| `gcoderag/generator.py` | Generate, validate, then retry with the validator's error list |
| `gcoderag/validator.py` | Parser + modal-state machine + 16 rules; CLI `python -m gcoderag.validator` |
| `gcoderag/evaluate.py` | Recall@k, MRR, and the keyword-accuracy answer evaluation |
| `app.py` | Gradio UI; rebuilds missing corpus/index data at startup |

### Validator rules

| Rule | Severity | Catches |
| --- | --- | --- |
| E001 | error | Stray characters / malformed words |
| E002 | error | Letter without a number, or an unparseable value (`X1.2.3`) |
| E003 | error | G/M code unknown for the dialect (built-in lists + every code in the corpus) |
| E004 | error | Two codes from one modal group in a block (`G0 G1`, `G90 G91`, `M8 M9`). In the `haas` dialect, canned cycles (G73, G76, G80-G89) are their own group as on Fanuc controls, so `G00 G80` is legal there |
| E005 | error | Feed move (G1/G2/G3/canned cycles) before any F word |
| E006 | error | G2/G3 without I, J, K or R |
| E007 | error | M6 with no T number given |
| E008 | error | M3/M4 with no S speed given |
| E009 | error | Tracked X/Y/Z position leaves the envelope (X, Y ±400 mm, Z ±250 mm; G91 and G20 handled) |
| E010 | error | F <= 0 |
| W101 | warning | No M2/M30 |
| W102 | warning | Feed move with the spindle off |
| W103 | warning | Rapid (G0) plunge below Z0 |
| W104 | warning | Haas program not wrapped in `%` or missing `O<number>` |
| W105 | warning | No G20/G21 units line |
| W106 | warning | `#` parameters or `[...]` expressions (line skipped, not validated) |

## How to run

Windows with Git Bash (the build machine). On macOS/Linux use `.venv/bin/python` instead of `.venv/Scripts/python.exe`.

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install --upgrade pip
.venv/Scripts/python.exe -m pip install -r requirements.txt

cp .env.example .env            # optional: add an LLM key; leave blank for retrieval-only mode
.venv/Scripts/python.exe scripts/check_env.py          # prints the mode, never the key

.venv/Scripts/python.exe scripts/build_corpus.py       # downloads docs -> data/corpus/ (SKIP_HAAS=1 skips Haas)
.venv/Scripts/python.exe scripts/build_index.py        # embeds 276 chunks -> data/index/ (~1 min on CPU)
.venv/Scripts/python.exe scripts/run_eval.py           # -> results/results.json, results/results.md
.venv/Scripts/python.exe scripts/validate_examples.py  # -> results/validator_report.json

.venv/Scripts/python.exe app.py                        # http://127.0.0.1:7860
```

`data/` is gitignored and is **not** in the repository. The Haas manual is copyrighted, so neither the raw pages nor
the processed corpus are redistributed. `build_corpus.py` and `build_index.py` rebuild everything from the public
sources, and `app.py` runs both automatically if `data/` is missing.

Command line:

```bash
.venv/Scripts/python.exe -m gcoderag.cli search "what does G4 do" --mode hybrid --k 5
.venv/Scripts/python.exe -m gcoderag.cli ask "which word sets the dwell time?"
.venv/Scripts/python.exe -m gcoderag.cli generate "drill four 5 mm holes on a 60 mm square" --dialect linuxcnc
.venv/Scripts/python.exe -m gcoderag.validator examples/valid_square_pocket.nc --dialect linuxcnc
```

Tests: `.venv/Scripts/python.exe -m pytest -q` runs 90+ offline tests in about 10 s (`FakeEmbedder`, `FakeLLM`, no
network). `RUN_SLOW=1` adds the real-embedding integration test.

**LLM configuration** (`.env`): `LLM_PROVIDER=anthropic` with `LLM_API_KEY` (model defaults to `claude-haiku-5-5`), or
`LLM_PROVIDER=openai_compatible` with `LLM_API_KEY`, `LLM_MODEL` and `LLM_BASE_URL` (Groq, Gemini's OpenAI endpoint,
Ollama, OpenAI). Without a key, *Ask the manual* shows the top 5 passages with links and *Generate G-code* is disabled.
*Validate G-code* never needs a key.

### Deploying to Hugging Face Spaces

This build has not been deployed (no `HF_TOKEN` was configured). To deploy it yourself:

1. Create a free account at huggingface.co and a **Write** token (Settings > Access Tokens).
2. `.venv/Scripts/python.exe -m pip install huggingface_hub` and then `.venv/Scripts/hf.exe auth login` (or put `HF_TOKEN=...` in `.env`).
3. Run `.venv/Scripts/python.exe scripts/deploy_hf.py`. It creates `<your-user>/gcode-rag` as a Gradio Space, uploads
   the code (never `data/`, `.env` or `.venv/`), and uses [deploy/README_HF.md](deploy/README_HF.md) as the Space card.
4. In the Space settings, add the variable `SKIP_HAAS=1` and, optionally, the secrets `LLM_PROVIDER`, `LLM_API_KEY`,
   `LLM_MODEL`, `LLM_BASE_URL`. On first start the Space downloads the LinuxCNC docs and builds the index (a few minutes
   on the free CPU tier). Without secrets it runs in retrieval-only mode.

## Evaluation

The evaluation set is [eval/questions.jsonl](eval/questions.jsonl): **56 questions** in 6 categories (code_lookup 21,
parameters 9, canned_cycles 8, syntax 8, modal_groups 5, safety_practice 5). **44 of the 56 do not contain the code
they ask about** ("How do I pause the program for two seconds?" rather than "What does G4 do?"), because machinists
often describe what they want to do instead of naming the code. Each question lists 1-3 gold document ids
(verified to exist in the corpus) and 1-3 answer keywords taken from the gold text. The gold ids were written before
the first evaluation run and were not changed afterwards.

Metrics, computed on distinct documents (chunks collapsed to their section first):

- **Recall@k**: fraction of questions with any gold document in the top k (k = 1, 5).
- **MRR**: mean of 1/rank of the first gold document within the top 10 (0 if absent).
- **Keyword accuracy** (needs an LLM key): fraction of answers that contain every answer keyword after lowercasing and
  code normalization. This is a cheap, deterministic proxy for correctness, not a human judgment. It can reject a
  correct answer that is worded differently, and it can accept a wrong answer that happens to contain the keywords.

Three configurations: `dense_baseline` (embeddings only), `bm25_only` (lexical only), `hybrid_rrf` (both, fused with
RRF k = 60, top 50 from each ranker).

## Results

56 questions, embedding model `BAAI/bge-small-en-v1.5`, corpus 150 documents / 276 chunks (LinuxCNC 150, Haas 0).
Copied from [results/results.md](results/results.md):

| config | recall@1 | recall@5 | MRR |
| --- | --- | --- | --- |
| dense_baseline | 0.5714 | 0.9464 | 0.6961 |
| bm25_only | 0.5714 | 0.8571 | 0.7008 |
| hybrid_rrf | 0.7500 | 0.9464 | 0.8363 |

Answer evaluation skipped: no LLM key configured. (`results.json` records `"answer_eval": {"status":
"skipped_no_llm_key"}`. Add a key to `.env` and re-run `scripts/run_eval.py` to fill in keyword accuracy and
`results/answers.jsonl`.)

**Why the numbers differ.** I split the per-question ranks in `results.json` (`per_question`) by whether the question
text names a G/M code (Recall@5 is 1.00 for all three configs on the code questions):

| config | names a code (n=12): R@1 / MRR | no code (n=44): R@1 / R@5 / MRR |
| --- | --- | --- |
| dense_baseline | 0.58 / 0.72 | 0.57 / 0.93 / 0.69 |
| bm25_only | 0.92 / 0.94 | 0.48 / 0.82 / 0.64 |
| hybrid_rrf | 1.00 / 1.00 | 0.68 / 0.93 / 0.79 |

- **BM25 wins on exact codes.** After `G01 -> G1` normalization, the token `g1` matches the section about G1 almost
  uniquely. Dense embeddings blur `G1`, `G10` and `G41` together, so dense ranks the right section first for only
  7 of the 12 code questions.
- **Dense wins on paraphrases.** "Pause", "stop the spindle" and "machine coordinates" share few tokens with the
  manual's wording, and that is where BM25 drops to 0.82 Recall@5.
- **Hybrid gets both.** Each ranker puts the right section first on 10 questions where the other does not. RRF keeps
  dense's recall (0.9464) and raises the first-hit rate from 57 % to 75 %. The best config is `hybrid_rrf`.
- **Remaining misses** (hybrid, not in the top 5): q050 "pause the program for two seconds" (G4 Dwell is outranked by
  M0/M1 *Program Pause* and M60 *Pallet Change Pause*, which use the word "pause"), q017 machine coordinates (G53, at
  rank 7), and q045 disabling overrides (M48/M49, at rank 8). q050 also shows a failure mode of RRF:
  `bm25_only` ranks G4 *first*, but G4 is not among dense's top 50 chunks, so it gets only one RRF term (about 1/61).
  M60 and M0/M1 are near the top of *both* lists and get two terms each (about 2/62), so they outrank it.

**Validator on the example programs** ([results/validator_report.json](results/validator_report.json)): all 7
examples match their expected outcome. The 3 valid programs (a LinuxCNC pocket, a peck-drilling pattern and a Haas
facing program) have 0 errors and 0 warnings. The 4 invalid ones are flagged with exactly the intended rules
(E005, E004, E003, and E006 + E009).

## Safety

Generated G-code is for learning and must be checked in a simulator (such as NC Viewer) and reviewed by a qualified machinist before it is run on a real machine.

The validator catches common, mechanical mistakes. It does **not** know your machine, fixture, stock, tool lengths,
work offsets or clamps. It does not simulate material removal or check collisions or cutter compensation geometry.
A program that passes it can still crash a machine. The same sentence is shown in the app footer.

## Decisions

The full log with timestamps is in [docs/DECISIONS.md](docs/DECISIONS.md). The main ones:

- **LinuxCNC only.** The Haas manual pages returned HTTP 403 to the scraper, so the corpus is the open LinuxCNC
  documentation. The Haas parser is still unit-tested on a fixture, and the validator supports a `haas` dialect
  (`%` wrapping, `O` program numbers, Haas-only codes such as G12/G13/G187/M97).
- **Hybrid retrieval with code normalization.** `G01`, `g1` and `G 01` all become `G1` in both the chunks and the
  queries. Later chunks of a section get the section title prepended before indexing, so the code name is always
  searchable.
- **The validator is deterministic Python, not an LLM.** Safety checks need to be reproducible and explainable.
  `E005`/`W102` are reported once per program so that the retry prompt stays short. The `haas` dialect keeps
  canned cycles in their own modal group, so the standard Haas safe-start line `G00 G17 G40 G49 G80 G90` passes.
- **Retrieval-only fallback.** The build ran without an LLM key, so every LLM path is tested with a scripted
  `FakeLLM`, and the app degrades gracefully instead of failing.
- **Honest evaluation.** The gold ids were fixed before the first run. Hybrid met the 0.75 Recall@5 target on the
  first run, so no retrieval tuning was done against the test set.

## Limitations

- Answer quality is unmeasured until an LLM key is added. Keyword accuracy is a proxy, not a human evaluation.
- The corpus covers LinuxCNC only. Haas questions are answered from LinuxCNC text, and the two dialects differ in
  places (for example canned-cycle details and the codes available).
- The validator does not interpret `#` parameters, expressions or o-code control flow (W106), and it does not check
  cutter compensation or tool-length geometry.
- The 56 questions were written by the builder, not collected from real users.

## License

The code is MIT licensed (see [LICENSE](LICENSE)). The documentation that the assistant answers from is
Copyright (c) 2000-2025 LinuxCNC.org and licensed under the GNU Free Documentation License 1.1 or later (stated in
`docs/src/common/overleaf.adoc` of the LinuxCNC repository). The repository does not redistribute that
documentation: `scripts/build_corpus.py` downloads it at build time, and every answer links back to the source
section. The one exception is the small test fixture `tests/fixtures/linuxcnc_snippet.adoc`, which keeps its GFDL
notice. No Haas manual content is included.
