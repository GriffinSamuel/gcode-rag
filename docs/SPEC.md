# Technical Specification: gcode-rag

This is the contract. `scripts/verify_done.py` checks the artifacts described here. Where this document
gives a name, path, key, or threshold, use it exactly.

## 1. Fixed constants

| Name | Value |
| --- | --- |
| Package name | `gcoderag` |
| Python | 3.10+ |
| Embedding model (primary) | `BAAI/bge-small-en-v1.5` (384 dims, normalized) |
| Embedding model (fallback) | `sentence-transformers/all-MiniLM-L6-v2` |
| Chunk target size | 250 words, max 400 words, overlap 40 words |
| Retrieval k | 10 chunks retrieved, then deduplicated to distinct documents |
| RRF constant | `k = 60`; `score(d) = sum over rankers r of 1 / (60 + rank_r(d))`, ranks start at 1 |
| Min corpus | 120 docs, 150 chunks, 60 distinct codes |
| Min eval set | 40 questions, 4 categories |
| Retrieval threshold | best config Recall@5 >= 0.75 |
| Answer threshold | keyword accuracy >= 0.60 (only when an LLM key exists) |
| Generator retries | 3 attempts total |
| Default envelope (mm) | X [-400, 400], Y [-400, 400], Z [-250, 250] |

## 2. Repository layout

```
gcode-rag/
  CLAUDE.md  PROGRESS.md  README.md  FINAL_REPORT.md  requirements.txt  pytest.ini  .env.example  .gitignore
  app.py                       # Gradio UI (section 8)
  gcoderag/
    __init__.py                # __version__ = "0.1.0"
    config.py                  # Paths dataclass + load_settings() (python-dotenv)
    ingest/__init__.py
    ingest/linuxcnc.py         # section 3.1, 3.3
    ingest/haas.py             # section 3.2
    ingest/chunker.py          # section 3.4
    textnorm.py                # normalize_codes(), tokenize() (section 4.3)
    embed.py                   # section 4.1
    index.py                   # section 4.2, 4.4
    llm.py                     # section 6.1
    rag.py                     # section 6.2
    generator.py               # section 6.3
    examples.py                # section 6.4: example library + few-shot retrieval
    genbench.py                # section 6.5: generation benchmark logic
    validator.py               # section 7 (also runnable: python -m gcoderag.validator)
    evaluate.py                # section 5
    cli.py                     # python -m gcoderag.cli {search,ask,generate,validate}
  scripts/
    check_env.py  build_corpus.py  build_index.py  run_eval.py  validate_examples.py  verify_done.py
    check_questions.py         # validates eval/questions.jsonl (docs/EVAL.md section 8)
    run_gen_eval.py            # generation benchmark (section 6.5)
  tests/
    fixtures/linuxcnc_snippet.adoc  fixtures/haas_snippet.html  fixtures/toy_corpus.jsonl
    test_smoke.py test_linuxcnc_parser.py test_chunker.py test_textnorm.py test_embed.py test_index.py
    test_evaluate.py test_llm.py test_rag.py test_validator.py test_generator.py test_app.py
    test_example_library.py test_genbench.py test_examples.py
  eval/questions.jsonl         # 177 questions with a dev/test split (docs/EVAL.md)
  eval/gen_requests.jsonl      # 30 generation requests: id, dialect, request (section 6.5)
  examples/*.nc                # validator examples (section 7.5)
  examples/library/*.nc        # 8 validator-clean few-shot programs (section 6.4)
  data/raw/linuxcnc/*.adoc  data/raw/haas/*.html  data/raw/haas/STATUS.json      (gitignored)
  data/corpus/docs.jsonl  data/corpus/chunks.jsonl  data/corpus/stats.json       (gitignored)
  data/index/embeddings.npy  data/index/meta.json                                (gitignored)
  results/results.json  results/results.md  results/validator_report.json  results/answers.jsonl  results/gen_results.json (committed)
  deploy/README_HF.md
  docs/SPEC.md  docs/EVAL.md  docs/DEPLOY.md  docs/DECISIONS.md
```

`gcoderag/config.py`:

```python
@dataclass(frozen=True)
class Paths:
    root: Path; data: Path; raw: Path; corpus: Path; index: Path; results: Path; eval: Path
def get_paths(root: Path | None = None) -> Paths   # root defaults to the repo root (parent of gcoderag/)
def load_settings() -> dict[str, str]              # loads .env via python-dotenv, returns os.environ view of the LLM_*/HF_*/EMBEDDING_MODEL keys
```

## 3. Corpus

### 3.1 LinuxCNC source (required, open documentation)

Download with `requests` (timeout 30 s, 2 retries, User-Agent `gcode-rag/0.1 (student portfolio project)`) from
`https://raw.githubusercontent.com/LinuxCNC/linuxcnc/master/docs/src/gcode/<file>`:

Required: `g-code.adoc`, `m-code.adoc`, `overview.adoc`.
Optional (skip silently on 404): `o-code.adoc`, `other-code.adoc`, `coordinates.adoc`, `tool-compensation.adoc`, `machining-center.adoc`, `rs274ngc.adoc`.

Save raw files to `data/raw/linuxcnc/<file>`. If the file already exists, do not re-download.

Public URL for citations: `https://linuxcnc.org/docs/stable/html/gcode/<file-stem>.html#<anchor>`
(for example `.../g-code.html#gcode:g0`). Do not fetch linuxcnc.org itself: its robots.txt disallows crawlers.

### 3.2 Haas source (best effort, copyrighted, local only)

Pages: `https://www.haascnc.com/service/online-operator-s-manuals/mill-operator-s-manual/mill---g-codes.html` and
`.../mill---m-codes.html`. Procedure:
1. Check `https://www.haascnc.com/robots.txt` with `urllib.robotparser`. If disallowed -> skip.
2. Download with the same requests settings, 1 request per second. On any HTTP error -> skip.
3. Sanity check: the G-code HTML must contain the text `G01` and `Linear Interpolation`; otherwise it is JS-rendered -> skip.
4. Parse with BeautifulSoup (`html.parser`). Each `h2`/`h3`/`h4` heading starts a document; its text is the heading
   plus all following sibling text until the next heading. Also parse the reference table rows (code, name, group)
   into one document per row titled `<code> <name>` with the text `"<code> <name> (Group <group>). See <url>."`.
5. Write `data/raw/haas/STATUS.json`: `{"status": "ok" | "skipped", "reason": "...", "pages": {...}, "docs": N}`.

Haas docs have `source: "haas"`, `doc_id: "haas:mill-gcodes:<slug>"` or `"haas:mill-mcodes:<slug>"` where `slug` is the
heading lowercased, non-alphanumerics -> `-`, collapsed, max 60 chars, de-duplicated with a numeric suffix.

### 3.3 AsciiDoc parsing -> documents

Section boundaries in LinuxCNC files are an anchor line followed by a level-2 heading:

```
[[gcode:g2-g3]]
== G2, G3 Arc Move(((G2, G3 Arc Move)))
```

Rules:
- A document starts at each `== ` heading (level exactly 2). The anchor is the nearest preceding `[[...]]` line
  (within 3 lines); if none, synthesize `sec-<slug>` from the heading.
- `doc_id = f"linuxcnc:{anchor}"` -> examples: `linuxcnc:gcode:g0`, `linuxcnc:mcode:m3-m4-m5`,
  `linuxcnc:gcode:modal-groups` (overview anchors are `gcode:` or `sec:`/`cap:` prefixed; keep them verbatim after `linuxcnc:`).
  If the same doc_id appears twice, append `-2`, `-3`.
- `title` = heading text with the `(((...)))` index term removed and whitespace collapsed.
- `codes` = all matches of `\b([GM])\s?0*(\d+(?:\.\d+)?)\b` in the heading, normalized (`G01` -> `G1`, `M03` -> `M3`),
  plus for "G10 L2" style headings the base code `G10`. Deduplicated, order preserved.
- `text` = the cleaned section body: drop `////` comment blocks, `image::` lines, `:attribute:` lines, `include::` lines,
  `(((...)))` terms, `[[anchors]]`, table markup `|===` fences (keep cell text, join cells with ` | `), and the `//`
  line comments. Keep `[source,ngc]` code blocks as plain text (they are examples a machinist wants). Convert
  `<<gcode:g1,G1>>` cross-references to `G1`. Collapse 3+ newlines to 2. Strip.
- Skip documents whose cleaned text is shorter than 40 characters.
- `url` per section 3.1; `section_path` = `"<file-stem> > <title>"`.
- Prepend `"<title>\n\n"` to `text` so the code name is always inside the embedded text.

Document record (`data/corpus/docs.jsonl`, one JSON object per line, UTF-8):

```json
{"doc_id": "linuxcnc:gcode:g4", "source": "linuxcnc", "title": "G4 Dwell", "codes": ["G4"],
 "url": "https://linuxcnc.org/docs/stable/html/gcode/g-code.html#gcode:g4",
 "section_path": "g-code > G4 Dwell", "text": "G4 Dwell\n\nG4 P- ..."}
```

Entry point: `python -m gcoderag.ingest.linuxcnc` downloads, parses, prints `docs: N` and writes nothing else
(the build script writes files). Function signatures:

```python
def download_sources(raw_dir: Path) -> list[Path]
def parse_adoc(text: str, file_stem: str) -> list[Doc]          # pure, unit-tested on the fixture
def load_all(raw_dir: Path) -> list[Doc]
```

### 3.4 Chunking

```python
def chunk_doc(doc: Doc, target_words: int = 250, max_words: int = 400, overlap_words: int = 40) -> list[Chunk]
```
- Split text into paragraphs (blank-line separated). Pack consecutive paragraphs while the total stays <= `target_words`.
- A single paragraph longer than `max_words` is split on sentence boundaries, then by words, with `overlap_words` overlap.
- `chunk_id = f"{doc_id}#{chunk_index}"` (0-based). Every chunk carries `doc_id, source, title, codes, url, chunk_index, text`.
- The first chunk of every doc starts with the title line (it is already in `text`).

`scripts/build_corpus.py` runs LinuxCNC (required) then Haas (best effort), writes `docs.jsonl`, `chunks.jsonl`,
and `stats.json` = `{"num_docs", "num_chunks", "sources": {"linuxcnc": n, "haas": n}, "distinct_codes": n, "built_at"}`,
and prints the stats.

## 4. Retrieval

### 4.1 `gcoderag/embed.py`

```python
class Embedder(Protocol):
    name: str
    dim: int
    def embed(self, texts: list[str]) -> np.ndarray      # shape (n, dim), float32, L2-normalized rows
class SentenceTransformerEmbedder(Embedder)             # model_name default from EMBEDDING_MODEL env or the primary model; batch_size 32
class FastEmbedEmbedder(Embedder)                       # same interface via fastembed.TextEmbedding
class FakeEmbedder(Embedder)                            # dim 64; deterministic: hash each token to a bucket, count, L2-normalize. Zero network.
def get_embedder(name: str | None = None) -> Embedder   # tries SentenceTransformer, then FastEmbed, raises RuntimeError with both errors
```

### 4.2 `gcoderag/index.py`

```python
@dataclass
class Hit: chunk_id: str; doc_id: str; score: float; text: str; title: str; url: str; codes: list[str]; source: str

class DenseIndex:
    @classmethod
    def build(cls, chunks: list[Chunk], embedder: Embedder) -> "DenseIndex"
    def save(self, index_dir: Path) -> None     # embeddings.npy (float32) + meta.json
    @classmethod
    def load(cls, index_dir: Path, chunks: list[Chunk]) -> "DenseIndex"
    def search(self, query_vec: np.ndarray, k: int) -> list[tuple[int, float]]   # (row, cosine) sorted desc

class BM25Index:
    def __init__(self, chunks: list[Chunk])      # rank_bm25.BM25Okapi over tokenize(normalize_codes(text))
    def search(self, query: str, k: int) -> list[tuple[int, float]]

class HybridRetriever:
    def __init__(self, chunks, dense: DenseIndex, bm25: BM25Index, embedder: Embedder)
    def retrieve(self, query: str, k: int = 10, mode: Literal["dense", "bm25", "hybrid"] = "hybrid") -> list[Hit]
    def retrieve_docs(self, query: str, k: int = 10, mode=...) -> list[str]   # distinct doc_ids in rank order
```
`meta.json`: `{"embedding_model": str, "dim": int, "num_vectors": int, "chunk_ids": [...], "built_at": iso8601}`.
`num_vectors` must equal `len(chunk_ids)` and the chunk count in `chunks.jsonl`.

### 4.3 `gcoderag/textnorm.py`

```python
CODE_RE = re.compile(r"\b([GgMm])\s?0*(\d+(?:\.\d+)?)\b")
def normalize_codes(text: str) -> str   # "g01" -> "G1", "M 03" -> "M3", "G38.2" unchanged, "G00" -> "G0"
def tokenize(text: str) -> list[str]    # lowercase; split on non [a-z0-9.]; drop empty; strip trailing '.'
```
Apply `normalize_codes` to chunk text before BM25 tokenization and before dense embedding, and to every query.

### 4.4 RRF

For `mode="hybrid"`: take top-50 from dense and top-50 from BM25, compute RRF with `k=60`, return top `k` chunks.
Ties broken by dense score. `retrieve_docs` deduplicates by `doc_id` keeping first occurrence.

### 4.5 `scripts/build_index.py`

Loads `chunks.jsonl`, builds `DenseIndex` with `get_embedder()`, saves to `data/index/`, prints model, dim, num_vectors, seconds.

### 4.6 Smoke queries (PROGRESS 2.3)

| Query | Expected doc in top 5 |
| --- | --- |
| `how do I pause the program for 2 seconds` | `linuxcnc:gcode:g4` |
| `rapid move` | `linuxcnc:gcode:g0` |
| `start the spindle clockwise` | `linuxcnc:mcode:m3-m4-m5` |
| `cut a clockwise arc` | `linuxcnc:gcode:g2-g3` |
| `end of program and rewind` | `linuxcnc:mcode:m2-m30` |
| `what does G01 do` | `linuxcnc:gcode:g1` |

`python -m gcoderag.cli search "<query>" [--mode hybrid] [--k 5]` prints rank, score, doc_id, title.

## 5. Evaluation (`gcoderag/evaluate.py`, `scripts/run_eval.py`)

### 5.1 Question file

`eval/questions.jsonl`, see `docs/EVAL.md` for the schema and writing rules.

### 5.2 Metrics (on distinct doc ids from `retrieve_docs(query, k=10, mode)`)

- `recall_at_1`, `recall_at_5`: fraction of questions where ANY gold doc id is within the first 1 / 5 distinct docs.
- `mrr`: mean over questions of `1 / rank` of the first gold doc in the top 10, else 0.
- Round to 4 decimals.

```python
def recall_at_k(ranked: list[str], gold: set[str], k: int) -> float          # 1.0 or 0.0
def reciprocal_rank(ranked: list[str], gold: set[str], max_rank: int = 10) -> float
def evaluate_config(questions, retriever, mode) -> dict
```

### 5.3 Configs (all three required, names exact)

| name | mode | description |
| --- | --- | --- |
| `dense_baseline` | dense | bge-small cosine only |
| `bm25_only` | bm25 | BM25 over normalized tokens |
| `hybrid_rrf` | hybrid | dense + BM25 fused with RRF k=60 |

### 5.4 Answer evaluation (only when an LLM is configured)

For each question: `rag.answer(question)` with `hybrid_rrf`; pass if, after `normalize_codes` and lowercasing, EVERY
string in `answer_keywords` appears in the answer text. `keyword_accuracy` = passes / n. Save every answer to
`results/answers.jsonl` (`{"id", "question", "answer", "citations", "passed"}`). Use `temperature=0`, `max_tokens=400`,
sleep 0.5 s between calls, retry 3x with backoff on API errors.

### 5.5 Output: `results/results.json`

```json
{
  "generated_at": "2026-10-12T03:21:00Z",
  "corpus": {"num_docs": 180, "num_chunks": 240, "sources": {"linuxcnc": 180, "haas": 0}},
  "embedding_model": "BAAI/bge-small-en-v1.5",
  "n_questions": 42,
  "configs": [
    {"name": "dense_baseline", "mode": "dense", "description": "...", "recall_at_1": 0.0, "recall_at_5": 0.0, "mrr": 0.0},
    {"name": "bm25_only", ...},
    {"name": "hybrid_rrf", ...}
  ],
  "best_config": "hybrid_rrf",
  "answer_eval": {"status": "done", "config": "hybrid_rrf", "provider": "anthropic", "model": "claude-haiku-5-5",
                  "n": 42, "keyword_accuracy": 0.81}
}
```
When no LLM: `"answer_eval": {"status": "skipped_no_llm_key"}`. `results/results.md` renders the same numbers as a
markdown table (this table is pasted into the README).

## 6. Generation

### 6.1 `gcoderag/llm.py`

```python
class LLM(Protocol):
    name: str
    def complete(self, system: str, user: str, max_tokens: int = 800, temperature: float = 0.0) -> str
class AnthropicLLM(LLM)            # anthropic SDK; model from LLM_MODEL (default "claude-haiku-5-5")
class OpenAICompatibleLLM(LLM)     # openai SDK with base_url=LLM_BASE_URL; works for Groq, Gemini (OpenAI endpoint), Ollama, OpenAI
class FakeLLM(LLM)                 # returns scripted responses from a list in order (repeats the last one); records calls
def get_llm_from_env() -> LLM | None   # LLM_PROVIDER in {"anthropic","openai_compatible"} AND LLM_API_KEY set -> client, else None
```
Env keys (see `.env.example`): `LLM_PROVIDER`, `LLM_API_KEY`, `LLM_MODEL`, `LLM_BASE_URL`. Accept `ANTHROPIC_API_KEY` as an
alias for `LLM_API_KEY` when provider is anthropic.

### 6.2 `gcoderag/rag.py`

```python
@dataclass
class Answer: text: str; citations: list[Hit]; grounded: bool; mode: Literal["llm", "retrieval-only"]
class RagPipeline:
    def __init__(self, retriever: HybridRetriever, llm: LLM | None)
    def answer(self, question: str, k: int = 5) -> Answer
```
- Retrieve `k=10` hybrid, dedupe to the first 5 distinct docs (use each doc's best chunk) -> passages `[1]..[5]`.
- If `llm is None`: `mode="retrieval-only"`, `text` = "Top passages from the documentation:" + the 5 passages with
  titles and URLs, `grounded=True`.
- Else system prompt: "You are a CNC programming assistant. Answer ONLY from the numbered passages. Cite passages as [1],
  [2]. Keep answers under 120 words. If the passages do not contain the answer, reply exactly: NOT_IN_DOCS: <one sentence
  saying what is missing>." User message: the passages (title, URL, text) then `Question: ...`.
- `grounded` = text does not start with `NOT_IN_DOCS` AND contains at least one `[n]` with `1 <= n <= 5`.
- `citations` = the hits whose `[n]` appear in the text (all 5 in retrieval-only mode).

### 6.3 `gcoderag/generator.py`

```python
@dataclass
class GenResult: program: str; report: ValidationReport; attempts: int; dialect: str; history: list[dict]
class GcodeGenerator:
    def __init__(self, retriever, llm: LLM, validator: GcodeValidator, library: ExampleLibrary | None = None)
    def generate(self, request: str, dialect: Literal["linuxcnc", "haas"] = "linuxcnc", max_attempts: int = 3) -> GenResult
```
- Retrieve 6 passages for the request (hybrid) to ground the codes used.
- System prompt states the dialect rules: units line (G21 or G20), G90, G17, tool change (T1 M6), spindle `S<rpm> M3`,
  feed `F` before the first G1/G2/G3, safe Z retract (G0 Z5 or higher) before XY rapids, `M5` then `M30` at the end;
  Haas: wrap with `%`, start with `O00001`; stay within the envelope; output ONLY the program inside a ```gcode fence.
- Extract the fenced block and run the validator. With errors: append the error list to the user message ("Fix these
  validation errors and return the complete corrected program:") and retry. With no errors but with warnings, and
  attempts remaining: send ONE more request ("The program is valid but has these warnings; fix them if possible and
  return the complete program: <warnings>"). `attempts` = number of LLM calls (at most `max_attempts`).
- The result is the best attempt by (number of errors, number of warnings), so a clean program is never lost to a
  worse retry. `GenResult.history` has one dict per LLM call with `attempt`, `program`, `ok`, `errors`, `warnings`
  (text) and `n_errors`, `n_warnings`.
- With a `library`, the user message includes the 2 most similar library programs under "Reference programs that
  passed validation:" (section 6.4), after the reference passages and before the request.
- If `llm is None`, raise `RuntimeError("LLM not configured")` (the app shows a friendly message instead).

### 6.4 `gcoderag/examples.py` and `examples/library/`

`examples/library/*.nc` holds 8 programs that validate with 0 errors and 0 warnings in their own dialect: LinuxCNC face
a block, square pocket, drill pattern (G81), peck drill (G83), rectangular contour with cutter compensation,
single-stroke engraving; Haas face a block and drill pattern. Each starts with `(dialect: ...)` and `(request: ...)`
header comments; the request text is what gets embedded.

```python
@dataclass(frozen=True)
class Example: name: str; dialect: str; request: str; program: str
def load_examples(directory: Path | None = None) -> list[Example]
class ExampleLibrary:
    def __init__(self, examples: list[Example], embedder: Embedder)        # embeds normalize_codes(request) once
    @classmethod
    def from_directory(cls, embedder, directory=None) -> ExampleLibrary
    def similar(self, request: str, k: int = 2, dialect: str | None = None) -> list[Example]   # cosine, best first
```
With `dialect`, same-dialect examples are ranked before the others. `app.py` and `cli generate` build the library with
the retriever's embedder.

### 6.5 Generation benchmark (`gcoderag/genbench.py`, `scripts/run_gen_eval.py`, `eval/gen_requests.jsonl`)

`eval/gen_requests.jsonl`: 30 requests, one JSON object per line (`id` g01-g30, `dialect`, `request`), covering sizes,
depths, tools, speeds, metric and inch, LinuxCNC (22) and Haas (8). `scripts/run_gen_eval.py` runs the generator on each
request (3 attempts max) and writes `results/gen_results.json`:

```json
{"runs": {"without_examples": {"label": "...", "generated_at": "...", "model": "...", "examples": false, "n": 30,
                               "first_attempt_ok_rate": 0.0, "final_ok_rate": 0.0, "mean_attempts": 0.0,
                               "mean_warnings_final": 0.0, "zero_warning_rate": 0.0, "feed_below_z0_rate": 0.0,
                               "requests": [{"id": "g01", "first_attempt_ok": true, "final_ok": true, "attempts": 1,
                                             "warnings_final": 0, "feed_below_z0": true, "errors_final": [],
                                             "program": "..."}]},
          "with_examples": {"...": "same shape, examples: true"}}}
```
`--no-examples` produces the `without_examples` run, the default produces `with_examples`; each run replaces only its
own entry. `first_attempt_ok` = no validator errors on attempt 1; `final_ok` = no errors in the returned program;
`warnings_final` = warnings in the returned program; `feed_below_z0` = the final program has a feed move (G1/G2/G3 or
a canned cycle) ending below Z0, i.e. it actually cuts. A request whose LLM call fails counts as not ok.

## 7. Validator (`gcoderag/validator.py`) - deterministic, no LLM, no network

### 7.1 Parsing

```python
@dataclass
class Word: letter: str; value: float; raw: str
@dataclass
class Block: line_no: int; raw: str; words: list[Word]; skipped: bool   # skipped: block-delete '/' or o-code line or '%'
def parse_program(text: str) -> list[Block]
```
- Strip `( ... )` comments and everything after `;`. Trim. `%` alone -> skipped block (remember it was seen for W104).
- Lines starting with `/` -> skipped (block delete). Lines matching `^[oO]\d*\s*(sub|endsub|call|if|elseif|else|endif|while|endwhile|do|break|continue|return|repeat|endrepeat)\b` or `^[oO]<` -> skipped (LinuxCNC o-codes). A bare `O\d+` (Haas program number) is a normal word.
- Words: regex `([A-Za-z])\s*([-+]?(?:\d+\.?\d*|\.\d+))`. Any leftover non-space characters -> E001. Letters outside
  `A-Z` set `ABCDEFGHIJKLMNOPQRSTUVWXYZ` cannot occur; `#` parameters (`#100=5`, `X#1`) -> treat the whole line as skipped with a W106 "parameters not validated" warning.
- `N` words are ignored for rules.

### 7.2 Known codes

`KNOWN_G_LINUXCNC`, `KNOWN_M_LINUXCNC`, `KNOWN_G_HAAS_EXTRA`, `KNOWN_M_HAAS_EXTRA` are module-level frozensets of
normalized codes (`"G1"`, `"G38.2"`, `"M30"`). The validator also accepts any code present in the corpus for the dialect
(`data/corpus/docs.jsonl` codes, loaded lazily if the file exists; otherwise built-ins only).

Built-in LinuxCNC G: `G0 G1 G2 G3 G4 G5 G5.1 G5.2 G5.3 G7 G8 G10 G17 G17.1 G18 G18.1 G19 G19.1 G20 G21 G28 G28.1 G30 G30.1 G33 G33.1
G38.2 G38.3 G38.4 G38.5 G40 G41 G41.1 G42 G42.1 G43 G43.1 G43.2 G49 G52 G53 G54 G55 G56 G57 G58 G59 G59.1 G59.2 G59.3 G61 G61.1 G64
G70 G71 G71.1 G71.2 G72 G72.1 G72.2 G73 G74 G76 G80 G81 G82 G83 G84 G85 G86 G87 G88 G89 G90 G90.1 G91 G91.1 G92 G92.1 G92.2 G92.3
G93 G94 G95 G96 G97 G98 G99`
Built-in LinuxCNC M: `M0 M1 M2 M3 M4 M5 M6 M7 M8 M9 M19 M30 M48 M49 M50 M51 M52 M53 M60 M61 M62 M63 M64 M65 M66 M67 M68 M70 M71 M72 M73 M98 M99`
plus `M100`-`M199` (user commands).
Haas extra G: `G9 G12 G13 G47 G50 G51 G60 G65 G100 G101 G103 G107 G110`-`G129 G136 G141 G143 G150 G153 G154 G174 G184 G187 G188 G234 G253 G254 G255 G266 G268 G269`
Haas extra M: `M10 M11 M12 M13 M16 M17 M18 M21 M22 M23 M24 M25 M26 M27 M28 M29 M31 M33 M34 M35 M36 M39 M41 M42 M46 M50 M51`-`M58 M59 M69 M75 M76 M77 M78 M79 M80 M81 M82 M83 M84 M85 M86 M88 M89 M95 M96 M97 M98 M99 M109 M130 M131 M138 M139 M158 M159`
For dialect `haas`, known = LinuxCNC built-ins + Haas extras (Haas uses the same core codes).

### 7.3 Modal groups (a block may contain at most one code from each group)

```
G motion:      G0 G1 G2 G3 G5 G5.1 G5.2 G33 G33.1 G38.2 G38.3 G38.4 G38.5 G73 G76 G80 G81 G82 G83 G84 G85 G86 G87 G88 G89 (+ Haas G12 G13 G150)
G plane:       G17 G18 G19 G17.1 G18.1 G19.1
G distance:    G90 G91
G arc dist:    G90.1 G91.1
G feed mode:   G93 G94 G95
G units:       G20 G21
G cutter comp: G40 G41 G42 G41.1 G42.1
G tool length: G43 G43.1 G43.2 G49
G coord sys:   G54 G55 G56 G57 G58 G59 G59.1 G59.2 G59.3 (+ Haas G110-G129 G154)
G return mode: G98 G99
G path ctrl:   G61 G61.1 G64
G spindle mode:G96 G97
G lathe mode:  G7 G8
M stopping:    M0 M1 M2 M30 M60
M spindle:     M3 M4 M5
M coolant:     only flag M9 together with M7 or M8 (M7 and M8 may be active together)
M override:    M48 M49
```

### 7.4 Rules

Errors (make `report.ok = False`):

| id | condition |
| --- | --- |
| E001 | Malformed word or stray characters in a block |
| E002 | Letter with no number, or a numeric value that is not parseable |
| E003 | Unknown G or M code for the dialect |
| E004 | Two codes from the same modal group in one block |
| E005 | A feed move (G1 G2 G3 G33 G73 G76 G81-G89, Haas G12 G13 G150) while no F has ever been set (F in the block counts) |
| E006 | G2/G3 without I, J, K, or R |
| E007 | M6 with no T in the block and no T previously seen |
| E008 | M3/M4 with no S in the block and no S previously seen |
| E009 | Any X/Y/Z position outside the envelope (track position; G91 adds, G90 sets; start at 0,0,0; axis words on G0/G1/G2/G3/canned cycles only) |
| E010 | F <= 0 |

Warnings (do not affect `ok`):

| id | condition |
| --- | --- |
| W101 | No M2/M30 anywhere |
| W102 | A feed move while spindle state is off (no M3/M4 active, or M5 active) |
| W103 | G0 that moves Z downward to a value below 0 (rapid plunge) |
| W104 | Haas dialect: program does not start with `%` and end with `%`, or no `O<number>` word |
| W105 | No G20/G21 anywhere |
| W106 | Line uses `#` parameters or expressions; skipped, not validated |

```python
@dataclass
class Issue: rule: str; line_no: int; message: str; raw: str; severity: Literal["error", "warning"]
@dataclass
class ValidationReport: errors: list[Issue]; warnings: list[Issue]; ok: bool; dialect: str; blocks: int
    def to_dict(self) -> dict
    def summary(self) -> str          # human-readable, one issue per line: "E005 line 12: ... | G1 X10"
class GcodeValidator:
    def __init__(self, dialect: str = "linuxcnc", envelope: dict[str, tuple[float, float]] | None = None, extra_codes: set[str] | None = None)
    def validate(self, text: str) -> ValidationReport
```
CLI: `python -m gcoderag.validator <file.nc> [--dialect linuxcnc|haas] [--json]` prints the summary (or JSON) and exits 0
when `ok` else 1.

### 7.5 Examples (`examples/`)

At least: `valid_square_pocket.nc`, `valid_drill_pattern.nc`, `valid_haas_face.nc` (Haas dialect, `%`/`O` wrapped),
`invalid_missing_feed.nc` (E005), `invalid_modal_conflict.nc` (E004), `invalid_unknown_code.nc` (E003). Put the intended
dialect in the first comment line: `(dialect: haas)` or `(dialect: linuxcnc)`. `scripts/validate_examples.py` runs each
file with its dialect and writes `results/validator_report.json`:

```json
{"generated_at": "...", "results": [
  {"file": "examples/valid_square_pocket.nc", "dialect": "linuxcnc", "expected": "valid", "ok": true, "errors": 0, "warnings": 1, "matches_expectation": true}
]}
```
`expected` comes from the filename prefix. Every entry must have `matches_expectation: true`.

## 8. App (`app.py`, Gradio)

```python
def build_app(pipeline: RagPipeline, generator: GcodeGenerator | None, validator_factory) -> gr.Blocks
if __name__ == "__main__": build_app(...).launch()
```
Tabs:
1. `Ask the manual`: question box, "Ask" button, answer markdown, citations table (title, URL), expandable retrieved passages. Retrieval-only mode shows a banner "No LLM key configured: showing retrieved passages only".
2. `Generate G-code`: request box, dialect radio, "Generate" button, program code box, validation report, attempts count. Disabled with a message when no LLM.
3. `Validate G-code`: paste area, dialect radio, "Validate" button, report. Always available.
Footer: the Safety sentence from CLAUDE.md section 8.
`tests/test_app.py` must build the Blocks with `FakeEmbedder`/`FakeLLM` on the toy corpus without calling `launch()`.
Index and corpus load lazily at startup; if `data/index/` is missing, `app.py` builds it (so a fresh deploy works).

## 9. Tests: minimum content

- `test_textnorm.py`: `G01`->`G1`, `M 03`->`M3`, `G38.2` unchanged, tokenization.
- `test_linuxcnc_parser.py`: the fixture snippet (3 sections incl. an anchor-less one and a `[source,ngc]` block) ->
  3 docs with correct ids, titles, codes, url; comment blocks removed.
- `test_chunker.py`: long doc splits with overlap; short doc stays one chunk; chunk ids.
- `test_embed.py`: `FakeEmbedder` deterministic, normalized, shape.
- `test_index.py`: toy corpus of 8 chunks, `FakeEmbedder`; dense/bm25/hybrid return the obviously matching chunk first;
  RRF math on a hand-computed example; save/load roundtrip (`tmp_path`).
- `test_evaluate.py`: recall/MRR on hand-built rankings; `evaluate_config` on the toy corpus.
- `test_llm.py`: `FakeLLM` scripting; `get_llm_from_env()` returns None with an empty env (`monkeypatch`).
- `test_rag.py`: grounded answer with citations; `NOT_IN_DOCS` path -> `grounded=False`; retrieval-only mode.
- `test_validator.py`: one test per rule E001-E010 and W101-W105, a valid program with zero errors, Haas `%` handling, o-code skipping, comment stripping, envelope tracking in G91.
- `test_generator.py`: invalid-then-valid scripted `FakeLLM` -> `attempts == 2`, `report.ok`.
- `test_app.py`: Blocks builds.
Target: >= 25 passing tests, total runtime < 60 s without `RUN_SLOW`.

## 10. Git hygiene

`.gitignore` already covers `.venv/`, `data/`, `.env`, `__pycache__/`, `.pytest_cache/`, `.claude/hooks/state/`.
Commit after each PROGRESS item. Never commit `.env`. `results/` IS committed (numbers are the portfolio evidence).
