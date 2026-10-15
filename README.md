# gcode-rag: a CNC G-code assistant with cited answers and validated generation

Ask machining questions and get answers grounded in the LinuxCNC G-code documentation, with a citation to the
exact manual section. Describe a part and get a G-code program that a deterministic validator has checked
(modal conflicts, missing feed, envelope limits, and more). If the validator finds errors, the program is sent back
to the model with the error list, up to 3 attempts. The system is measured on a 177-question evaluation set
with a dev/test split, comparing three retrieval configurations.

- **Retrieval:** dense embeddings (`BAAI/bge-small-en-v1.5`) + BM25, fused with Reciprocal Rank Fusion (k = 60)
- **Generation:** pluggable LLM (Anthropic, or any OpenAI-compatible endpoint such as Groq, Gemini or Ollama)
- **Validation:** rule-based Python with 16 rules, no LLM in the loop, 31 unit tests
- **UI:** Gradio app with three tabs: *Ask the manual*, *Generate G-code*, *Validate G-code*

Headline numbers (from [results/results.json](results/results.json)): on the **held-out test split** (106 questions) the local model's answers contain the expected keywords **90.6 %** of the time (dev 83.1 %, all 177 questions 87.6 %). Hybrid retrieval puts a correct manual section in the top 5 for **94.3 %** of the 177 questions (Recall@5 0.9435) and ranks it first for **74.0 %** (Recall@1 0.7401, MRR 0.8278).

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
| `gcoderag/generator.py` | Generate, validate, retry with the error list, then one more request to fix any warnings; returns the best attempt |
| `gcoderag/examples.py` | Library of 8 validator-clean programs (`examples/library/`); the 2 most similar to a request go into the generation prompt |
| `gcoderag/genbench.py` | Generation benchmark logic (`scripts/run_gen_eval.py`, `eval/gen_requests.jsonl`) |
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

cp .env.example .env            # optional: add an LLM; leave the key blank for retrieval-only mode
.venv/Scripts/python.exe scripts/check_env.py          # prints the mode, never the key

.venv/Scripts/python.exe scripts/build_corpus.py       # downloads docs -> data/corpus/ (SKIP_HAAS=1 skips Haas)
.venv/Scripts/python.exe scripts/build_index.py        # embeds 276 chunks -> data/index/ (~1 min on CPU)
.venv/Scripts/python.exe scripts/check_questions.py    # validates eval/questions.jsonl (must print ALL CHECKS OK)
.venv/Scripts/python.exe scripts/run_eval.py           # -> results/results.json, results/results.md (--split dev|test|all)
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

**Free local LLM with Ollama** (what produced the answer evaluation above). Install [Ollama](https://ollama.com),
then:

```bash
ollama pull qwen2.5-coder:7b
ollama serve                    # skip if the Ollama tray app is already running (it serves on port 11434)
```

and put these four lines in `.env`:

```
LLM_PROVIDER=openai_compatible
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=qwen2.5-coder:7b
LLM_API_KEY=ollama
```

(`LLM_API_KEY` is a dummy value; Ollama ignores it.) Check with `scripts/check_env.py`, then run
`scripts/run_eval.py` to fill in `answer_eval` and `results/answers.jsonl`.

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

The evaluation set is [eval/questions.jsonl](eval/questions.jsonl): **177 questions** in 6 categories (code_lookup 37, parameters 40, modal_groups 15, syntax 49, canned_cycles 16, safety_practice 20).
**164 of the 177 do not contain the code they ask about** ("How do I pause the program for two seconds?" rather
than "What does G4 do?"), because machinists often describe what they want to do instead of naming the code. 28
questions need two or more documents. Each question lists 1-3 gold document ids (verified to exist in the corpus), 1-3
answer keywords taken from the gold text, and a `split`. The set was built from the corpus: the first 56 questions were written before any evaluation run, and the other 121 were written to cover the 84 documents that had no question yet (all 150 of the 150 documents are now the gold answer for at least one question), then checked by `scripts/check_questions.py` (schema, unique ids, gold ids exist, every keyword appears as a whole word in the gold text, category minimums, no near-duplicates, split fraction). No retrieval or answer result was computed for the new questions before they were written.

**Dev/test split.** Every question is `dev` (40 %, 71 questions) or `test` (60 %, 106), assigned
at random with `random.seed(42)`. Dev is for tuning; test is looked at once, at the end. `scripts/run_eval.py --split
dev` runs the answer evaluation on dev only (and prints only dev numbers); `--split all` is the final run.

Metrics, computed on distinct documents (chunks collapsed to their section first):

- **Recall@k**: fraction of questions with any gold document in the top k (k = 1, 5).
- **MRR**: mean of 1/rank of the first gold document within the top 10 (0 if absent).
- **Keyword accuracy** (needs an LLM): fraction of answers that contain every answer keyword as a whole word after
  lowercasing and code normalization. This is a cheap, deterministic proxy for correctness, not a human judgment. It
  can reject a correct answer that is worded differently, and it can accept a wrong answer that happens to contain the
  keywords.
- **Citation rate**: fraction of answers with at least one attached source (see Results for what that does and does
  not show).

Three retrieval configurations: `dense_baseline` (embeddings only), `bm25_only` (lexical only), `hybrid_rrf` (both,
fused with RRF k = 60, top 50 from each ranker). Answers always use `hybrid_rrf`.

## Results

177 questions (71 dev, 106 test), embedding model `BAAI/bge-small-en-v1.5`, corpus
150 documents / 276 chunks (LinuxCNC 150,
Haas 0). Copied from [results/results.json](results/results.json).

**Answer evaluation** with `qwen2.5-coder:7b` (`openai_compatible`, Ollama), config `hybrid_rrf`:

| split | n | keyword accuracy | citation rate |
| --- | --- | --- | --- |
| dev (used for tuning) | 71 | 0.8310 | 0.8732 |
| **test (held out, the headline)** | 106 | **0.9057** | 0.9057 |
| all | 177 | 0.8757 | 0.8927 |

Generated answers come from qwen2.5-coder:7b running locally through Ollama; no paid API is required. Tuning used only the dev split. I tried three changes, each scored on dev alone, and none beat the baseline (a longer prompt asking for the exact code and word letters 0.8169, six passages with the best one last 0.8169, section headings plus the question repeated 0.8028, baseline 0.8310), so none was kept. The test split was run once, at the end. Test came out higher than dev; with a random split of this size that gap is within noise, and since no tuning change was kept the test number was not tuned on.

The citation rate is deterministic: every answer that is not a `NOT_IN_DOCS` refusal gets a source attached (the model's `SOURCES:` line, else its `[n]` markers, else passages whose codes the answer mentions, else the top passage). So it equals 1 minus the refusal rate (19 of 177 answers were refusals) and shows that a source is attached, not that the model chose it correctly. 22 of the 177 answers missed a keyword (per-question answers are in
[results/answers.jsonl](results/answers.jsonl)).

**Retrieval** (all 177 questions):

| config | recall@1 | recall@5 | MRR |
| --- | --- | --- | --- |
| dense_baseline | 0.6893 | 0.9435 | 0.7880 |
| bm25_only | 0.6215 | 0.8757 | 0.7314 |
| hybrid_rrf | 0.7401 | 0.9435 | 0.8278 |

and per split:

| split | n | config | recall@1 | recall@5 | MRR |
| --- | --- | --- | --- | --- | --- |
| dev | 71 | dense_baseline | 0.6479 | 0.9577 | 0.7697 |
| dev | 71 | bm25_only | 0.6197 | 0.8732 | 0.7304 |
| dev | 71 | hybrid_rrf | 0.7324 | 0.9296 | 0.8176 |
| test | 106 | dense_baseline | 0.7170 | 0.9340 | 0.8002 |
| test | 106 | bm25_only | 0.6226 | 0.8774 | 0.7321 |
| test | 106 | hybrid_rrf | 0.7453 | 0.9528 | 0.8347 |

**Why the numbers differ.** I split the per-question ranks in `results.json` (`per_question`) by whether the question
text names a G/M code:

| config | names a code (n=13): R@1 / R@5 / MRR | no code (n=164): R@1 / R@5 / MRR |
| --- | --- | --- |
| dense_baseline | 0.62 / 1.00 / 0.74 | 0.70 / 0.94 / 0.79 |
| bm25_only | 0.92 / 1.00 / 0.94 | 0.60 / 0.87 / 0.71 |
| hybrid_rrf | 1.00 / 1.00 / 1.00 | 0.72 / 0.94 / 0.81 |

- **BM25 wins on exact codes.** After `G01 -> G1` normalization, the token `g1` matches the section about G1 almost
  uniquely. Dense embeddings blur `G1`, `G10` and `G41` together, so on the 13 questions that name a code
  dense ranks the right section first 62% of the time and BM25 92%.
- **Dense is better on paraphrases.** Most questions are phrased without a code ("stop the spindle", "machine
  coordinates"), which share few tokens with the manual's wording; on those 164 questions Recall@5 is
  0.94 for dense and 0.87 for BM25.
- **Hybrid wins at the first rank, not at Recall@5.** Dense ranks the right section first on 33 questions
  where BM25 does not, and BM25 on 21 where dense does not. Fusing them gets the right section first on
  131 of 177: Recall@1 0.7401 vs 0.6893 (dense) and 0.6215
  (BM25), MRR 0.8278 vs 0.7880 and 0.7314. For Recall@5 hybrid only ties dense overall
  (0.9435 each); on dev dense is higher (0.9577 vs 0.9296) and on
  test hybrid is higher (0.9528 vs 0.9340). The best config by Recall@5 and MRR
  is `hybrid_rrf`.
- **Remaining misses** (hybrid, not in the top 5): 10 of 177 questions (q017, q045, q050, q099, q100, q121, q151, q152, q155, q158). The
  clearest case is q050, "pause the program for two seconds": G4 Dwell is outranked by M0/M1 *Program Pause* and M60
  *Pallet Change Pause*, which use the word "pause". It also shows a failure mode of RRF: `bm25_only` ranks G4 *first*,
  but G4 is not among dense's top 50 chunks, so it gets only one RRF term (about 1/61), while M60 and M0/M1 are near the
  top of *both* lists and get two terms each (about 2/62).

**Validator on the example programs** ([results/validator_report.json](results/validator_report.json)): all 7
examples match their expected outcome. The 3 valid programs (a LinuxCNC pocket, a peck-drilling pattern and a Haas
facing program) have 0 errors and 0 warnings. The 4 invalid ones are flagged with exactly the intended rules
(E005, E004, E003, and E006 + E009).

### Generation benchmark

30 requests ([eval/gen_requests.jsonl](eval/gen_requests.jsonl): 22 LinuxCNC and 8 Haas, metric and inch, different
sizes, depths, tools and speeds) are sent to the generator with `qwen2.5-coder:7b` (up to 3 attempts, then the validator
decides). One run without the example library (`--no-examples`) and one with it (the 2 most similar of the 8 programs in
`examples/library/` are added to the prompt). Results: [results/gen_results.json](results/gen_results.json).

| examples in prompt | n | first_attempt_ok | final_ok | mean attempts | mean warnings | zero-warning rate | cuts below Z0 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| no | 30 | 0.7667 | 0.9667 | 1.6667 | 0.4000 | 0.6000 | 0.9667 |
| yes (2 similar programs) | 30 | 0.9333 | 1.0000 | 1.0667 | 0.0000 | 1.0000 | 1.0000 |

Columns: **first_attempt_ok** is the share of requests whose very first program had no validator errors;
**final_ok** the share whose returned program has no errors after the retries; **mean attempts** the average number of
LLM calls (a warning-fix request counts as an attempt); **mean warnings** the average number of validator warnings left
in the returned program and **zero-warning rate** the share that is completely clean; **cuts below Z0** the share whose
final program has a feed move (or canned cycle) that ends below Z0, i.e. it actually cuts material.

First-attempt pass rate went from **0.7667** to **0.9333** with
examples. Without examples, the leftover warnings were {'W103': 1, 'W102': 3, 'W104': 8} (W104 is "Haas program not wrapped in `%` with an
`O` number": all 8 Haas requests kept it even after the warning-fix request), 1 request
failed outright (g21) and 1 program never cut (g07). With examples
every program passed after the retries, with no warnings left, and the only first-attempt failures were
g13, g14.

How far to trust this: the validator checks rules (feed set, modal groups, envelope, Haas wrapping, and so on), not
geometry, so "ok" does not mean the part is the right size; the benchmark is 30 requests from one local 7B model, one
run each; and the 8 library programs cover the same kinds of job as the requests (face, pocket, drill, peck drill,
contour, engraving), so the gain is for requests *like* the library, and the examples mostly teach the model the exact
program format (for Haas: the `%`/`O` wrapper).

Run it yourself (needs an LLM, see `scripts/check_env.py`):

```bash
.venv/Scripts/python.exe scripts/run_gen_eval.py --no-examples   # run "without_examples"
.venv/Scripts/python.exe scripts/run_gen_eval.py                 # run "with_examples"
```

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
- **Retrieval-only fallback.** Every LLM path is tested with a scripted `FakeLLM`, and the app degrades gracefully
  to showing cited passages when no LLM is configured. The measured answer evaluation uses a free local model
  (`qwen2.5-coder:7b` via Ollama), so no paid API is needed.
- **Honest evaluation.** The gold ids and keywords are fixed before results are computed. Tuning used only the dev split and kept nothing, and the held-out test split is run once at the end.

## Limitations

- Keyword accuracy is a proxy, not a human evaluation, and the citation rate only shows that a source is attached (19 of 177 answers were `NOT_IN_DOCS` refusals).
- The corpus covers LinuxCNC only. Haas questions are answered from LinuxCNC text, and the two dialects differ in
  places (for example canned-cycle details and the codes available).
- The validator does not interpret `#` parameters, expressions or o-code control flow (W106), and it does not check
  cutter compensation or tool-length geometry.
- The 177 questions were written by the builder, not collected from real users, and the dev/test split is random, so the two halves differ by chance (test scored higher than dev).

## License

The code is MIT licensed (see [LICENSE](LICENSE)). The documentation that the assistant answers from is
Copyright (c) 2000-2025 LinuxCNC.org and licensed under the GNU Free Documentation License 1.1 or later (stated in
`docs/src/common/overleaf.adoc` of the LinuxCNC repository). The repository does not redistribute that
documentation: `scripts/build_corpus.py` downloads it at build time, and every answer links back to the source
section. The one exception is the small test fixture `tests/fixtures/linuxcnc_snippet.adoc`, which keeps its GFDL
notice. No Haas manual content is included.
