# Evaluation Design

The evaluation is what makes this project more than a tutorial RAG. Every number in the README comes from
`results/results.json`, which `scripts/run_eval.py` produces from `eval/questions.jsonl`.

## 1. Question file schema (`eval/questions.jsonl`, one JSON object per line)

```json
{"id": "q001", "category": "code_lookup", "question": "What does G4 do and which word sets the dwell time?",
 "gold_doc_ids": ["linuxcnc:gcode:g4"], "answer_keywords": ["dwell", "P"]}
```

| field | rules |
| --- | --- |
| `id` | unique, `q` + 3 digits |
| `category` | one of: `code_lookup`, `parameters`, `modal_groups`, `syntax`, `canned_cycles`, `safety_practice`, `haas_specific` |
| `question` | natural machinist phrasing; vary vocabulary; at least 10 questions must NOT contain the code they target (e.g. "pause the program for two seconds" instead of "what does G4 do") |
| `gold_doc_ids` | 1-3 ids that MUST exist in `data/corpus/docs.jsonl`; a question is "answerable" from any one of them |
| `answer_keywords` | 1-3 short, robust strings that any correct answer must contain after lowercasing and code normalization (`G01`->`G1`); prefer the code name or a single unambiguous noun; never a full sentence. Matching is by whole word: `p` does not match `program`, `g1` does not match `g10` |
| `split` | `dev` (about 40 %, used for tuning prompts and settings) or `test` (about 60 %, held out; run once at the end). Assigned at random with `random.seed(42)` over all questions |

Example with the split field:

```json
{"id": "q057", "category": "code_lookup", "question": "How do I program a cubic spline move in the XY plane?",
 "gold_doc_ids": ["linuxcnc:gcode:g5"], "answer_keywords": ["spline"], "split": "test"}
```

Requirements enforced by `scripts/verify_done.py`: >= 40 questions, >= 4 distinct categories, unique ids, every
gold id present in the corpus, every record has all five fields with non-empty lists.

## 2. How to write the set (do this AFTER the corpus exists)

1. Open `data/corpus/docs.jsonl` and list the doc ids and titles (`python -m gcoderag.cli docs` is a convenient helper
   if you add it; otherwise a one-liner).
2. Cover the core codes every machinist uses: G0 G1 G2/G3 G4 G17-19 G20/21 G28 G40-42 G43/49 G53 G54-59 G80-89 G90/91
   G93-95 G98/99, M0/M1 M2/M30 M3/M4/M5 M6 M7/M8/M9, plus overview sections (format of a line, modal groups, comments,
   order of execution, best practices) and o-codes if that file downloaded.
3. If Haas docs exist, add >= 5 `haas_specific` questions (e.g. G12/G13 circular pocket, G150 pocket milling,
   G187 accuracy control, M97 local subprogram call) with Haas gold ids.
4. Write the keywords by reading the gold doc text: pick words that appear there and that a paraphrased correct answer
   would still contain.
5. Mix difficulty: ~15 direct lookups, ~15 paraphrased/no-code questions, ~10 parameter/behavior questions.

## 3. Seed questions (valid for the LinuxCNC corpus; verify ids after the build)

```json
{"id": "q001", "category": "code_lookup", "question": "What does G4 do and which word sets the dwell time?", "gold_doc_ids": ["linuxcnc:gcode:g4"], "answer_keywords": ["dwell", "p"]}
{"id": "q002", "category": "code_lookup", "question": "How do I make a rapid positioning move?", "gold_doc_ids": ["linuxcnc:gcode:g0"], "answer_keywords": ["g0"]}
{"id": "q003", "category": "parameters", "question": "Which words can I use to define the center of an arc with G2 or G3?", "gold_doc_ids": ["linuxcnc:gcode:g2-g3"], "answer_keywords": ["i", "j"]}
{"id": "q004", "category": "code_lookup", "question": "I want the spindle to turn clockwise. Which M code do I use?", "gold_doc_ids": ["linuxcnc:mcode:m3-m4-m5"], "answer_keywords": ["m3"]}
{"id": "q005", "category": "code_lookup", "question": "How do I end the program and rewind to the start?", "gold_doc_ids": ["linuxcnc:mcode:m2-m30"], "answer_keywords": ["m30"]}
{"id": "q006", "category": "code_lookup", "question": "Which code performs a tool change?", "gold_doc_ids": ["linuxcnc:mcode:m6"], "answer_keywords": ["m6"]}
{"id": "q007", "category": "code_lookup", "question": "What is the difference between M0 and M1?", "gold_doc_ids": ["linuxcnc:mcode:m0-m1"], "answer_keywords": ["optional"]}
{"id": "q008", "category": "parameters", "question": "What does the F word mean in a feed move at programmed feed rate?", "gold_doc_ids": ["linuxcnc:gcode:g1"], "answer_keywords": ["feed"]}
```

Anchors seen in the LinuxCNC sources (ids are `linuxcnc:` + anchor): `gcode:g0`, `gcode:g1`, `gcode:g2-g3`, `gcode:g4`,
`gcode:g5`, `gcode:g5.1`, `gcode:g5.2-g5.3`, `gcode:g7`, `gcode:g8`, `gcode:g10-l0`, `gcode:g10-l1`, `gcode:g10-l2`,
`gcode:g10-l10`, `gcode:g10-l11`, `gcode:g10-l20`, `mcode:m0-m1`, `mcode:m2-m30`, `mcode:m60`, `mcode:m3-m4-m5`,
`mcode:m6`. All others: take them from `docs.jsonl`, never guess.

## 4. Metrics

- **Recall@k** (k = 1, 5): a question counts as a hit when any gold doc id is among the first k DISTINCT doc ids
  returned (chunks are collapsed to documents first). Reported as a fraction of questions.
- **MRR**: mean of 1/rank of the first gold doc within the top 10 distinct docs; 0 when absent.
- **Keyword accuracy** (LLM only): fraction of questions whose generated answer contains all `answer_keywords`
  after lowercasing + code normalization. It is a cheap, deterministic proxy for correctness; say so in the README.

## 5. Configurations compared

`dense_baseline` (embeddings only), `bm25_only` (lexical only), `hybrid_rrf` (RRF fusion, k=60). The README must
discuss WHY the numbers differ (e.g. BM25 is strong on exact code tokens like `G38.2`, dense is strong on paraphrases,
hybrid gets both). If `hybrid_rrf` is not the best config, say so and explain; do not hide it.

## 6. Results tables

`results/results.md` (generated) example:

```
| config | recall@1 | recall@5 | MRR |
| --- | --- | --- | --- |
| dense_baseline | 0.62 | 0.88 | 0.72 |
| bm25_only | 0.57 | 0.81 | 0.66 |
| hybrid_rrf | 0.69 | 0.93 | 0.78 |
```
Copy it into README `## Results` together with `n_questions`, the embedding model, corpus size, and the answer
evaluation line (or the sentence "Answer evaluation skipped: no LLM key configured").

## 7. Dev/test split and how to use it

Tuning anything (the system prompt, the number of passages, retrieval settings) on all questions would make the final
score partly measure how well the system was fitted to those exact questions. So the questions are split:

- **dev** (about 40 %): the only questions you may look at while tuning. `run_eval.py --split dev` runs the answer
  evaluation on dev only and prints only dev numbers.
- **test** (about 60 %): held out. Run `run_eval.py --split all` once at the end; the **test** numbers in
  `results.json` (`by_split.test`, `answer_eval.by_split.test`) are the headline.

Retrieval metrics are cheap and are always computed for every question, overall and per split (`by_split`). If a
change is kept only when dev accuracy goes up and rejected otherwise, test never influences a decision. Record every
tuning iteration (the change and its dev number) in `docs/DECISIONS.md`.

`results.json` fields added by the split: `split_evaluated`, `by_split.{dev,test}.{n,configs,answer_eval}`,
`answer_eval.by_split`, and `citation_rate` (overall in `answer_eval` and per split). The citation rate is the share of
answers with at least one attached source; because citations are attached deterministically (see `gcoderag/rag.py`), it
equals 1 minus the refusal rate.

## 8. Checking the question set: `scripts/check_questions.py`

```bash
.venv/Scripts/python.exe scripts/check_questions.py
```

Prints one OK/FAIL line per check and ends with `ALL CHECKS OK` (exit 0) or `<n> CHECK(S) FAILED` (exit 1). It checks:
fields and types (1-3 gold ids and 1-3 keywords), unique `q###` ids, allowed categories, every gold id exists in
`data/corpus/docs.jsonl`, every keyword occurs as a whole word in at least one gold document's text (after lowercasing
and code normalization), at least 160 questions, at least 10 questions in each of code_lookup, parameters,
modal_groups, syntax, canned_cycles and safety_practice, at least 40 % of questions that do not name a G/M code, at least
5 questions with two or more gold documents, no near-duplicate questions (word-set Jaccard >= 0.8), a `split` on every
question, and a dev share between 35 % and 45 %. Run it after every edit to `eval/questions.jsonl`.
