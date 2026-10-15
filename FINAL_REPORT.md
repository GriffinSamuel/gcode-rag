# Final report: gcode-rag

Read this before an interview. It tells you what the project is, which numbers to quote, what to admit, and what to
demo in five minutes.

## What was built

**One-sentence pitch:** "A RAG assistant for CNC machinists. It answers G-code questions with citations to the exact
section of the LinuxCNC manual. It generates G-code programs and checks them with a deterministic validator, and
when the validator finds errors it sends them back to the model for another attempt. I measured it on a
177-question set with a held-out test split."

The parts, and the one thing worth saying about each:

1. **Corpus.** The 9 LinuxCNC AsciiDoc files are parsed into 150 sections (one per `==` heading), then split into
   276 chunks (250-word target, median 191 words). Every chunk keeps a citation URL with the section anchor. I also wrote a Haas
   scraper with a robots.txt check and rate limiting. Haas returned HTTP 403, so I documented the skip instead of
   working around it.
2. **Retrieval.** `bge-small-en-v1.5` dense embeddings plus BM25, fused with Reciprocal Rank Fusion (k = 60). The
   key detail: `G01`, `g1` and `G 01` are all normalized to `G1` in both the documents and the queries. Without that,
   BM25 misses exact-code matches.
3. **Grounded answers.** The model may only use the five numbered passages and must cite them as `[n]`. If the
   passages do not contain the answer, it replies `NOT_IN_DOCS` instead of guessing.
4. **Validator.** About 400 lines of plain Python that track modal state (motion mode, G90/G91, units, feed, tool,
   spindle, position). It has 10 error rules (missing feed, conflicting modal codes, arcs without a centre, moves
   outside the envelope...) and 6 warnings. There is no LLM in it, so the result is reproducible and every flag can be
   explained line by line.
5. **Generate-validate-retry.** The model writes a program, the validator checks it, and up to 2 retries get the
   exact error lines back.
6. **A bug worth telling:** taken literally, the spec's "nearest anchor within 3 lines" rule gives the first
   section of each file the *chapter* anchor, because the lookback crosses the level-1 heading (`[[cha:g-codes]]`,
   `= G-Codes`, blank line, `== Conventions`). The parser stops the lookback at any heading, and a parser test pins
   it (`Conventions` gets `sec-conventions`). It is a small bug, but a wrong anchor produces a wrong
   citation link, which is exactly the error a RAG system must not make.
7. **Gradio app** with three tabs. It still works without an LLM (retrieval-only) and runs against a free local model through Ollama.
8. **Tests:** 95 offline tests that run in about 10 seconds, using a fake embedder and a scripted fake LLM.

## Results

Quote these (from `results/results.json`, 177 questions, 164 of them phrased without naming the code):

| split | n | keyword accuracy | citation rate |
| --- | --- | --- | --- |
| dev (used for tuning) | 71 | 0.8310 | 0.8732 |
| **test (held out, the headline)** | 106 | **0.9057** | 0.9057 |
| all | 177 | 0.8757 | 0.8927 |

| config | recall@1 | recall@5 | MRR |
| --- | --- | --- | --- |
| dense_baseline | 0.6893 | 0.9435 | 0.7880 |
| bm25_only | 0.6215 | 0.8757 | 0.7314 |
| hybrid_rrf | 0.7401 | 0.9435 | 0.8278 |

- **The line to remember:** "On 106 held-out test questions the local 7B model, running free through
  Ollama, gets the expected keywords 90.6 % of the time, and retrieval finds a correct manual
  section in the top 5 for 94.3 % of all 177 questions."
- **Why the dev/test split matters (say it like this):** "I split the questions into two piles. I was allowed to
  peek at the dev pile while I tried to improve the answers, and I only counted the test pile once at the end, like a
  final exam the model never saw while I was changing things. If I had tuned on all the questions, my score would
  partly measure how well I fitted those exact questions, not how well it works on new ones." Tuning used only the dev split. I tried three changes, each scored on dev alone, and none beat the baseline (a longer prompt asking for the exact code and word letters 0.8169, six passages with the best one last 0.8169, section headings plus the question repeated 0.8028, baseline 0.8310), so none was kept. The test split was run once, at the end. Test came out higher than dev; with a random split of this size that gap is within noise, and since no tuning change was kept the test number was not tuned on.
- **Why hybrid wins (this is the interesting part):** on the 13 questions that name a code, BM25 ranks the
  right section first 92% of the time and dense only 62%, because
  embeddings blur `G1`/`G10`/`G41`; on the 164 paraphrased questions dense reaches 0.94
  Recall@5 and BM25 0.87. Fusing them (RRF) puts the right section first more often
  (Recall@1 0.7401 vs 0.6893 dense and 0.6215 BM25), but Recall@5 only ties
  dense (0.9435); be ready to say that plainly.
- **Validator:** all 7 example programs are classified correctly. The 3 valid ones have zero findings, and each of
  the 4 invalid ones triggers exactly the intended rules.
- **Integrity point:** the gold ids and keywords were fixed before any result was computed for them, tuning used only the dev split, and none of the three tuning changes was kept, so the test number was not tuned on. `scripts/check_questions.py` checks the whole set.

## Limitations

Raise these yourself before you are asked:

- **Answer accuracy is a proxy.** Keyword accuracy is 0.8757 overall with `qwen2.5-coder:7b` running locally through Ollama (no paid API). It checks that the expected keywords appear, not that the answer is correct. The citation rate is deterministic: every answer that is not a `NOT_IN_DOCS` refusal gets a source attached (the model's `SOURCES:` line, else its `[n]` markers, else passages whose codes the answer mentions, else the top passage). So it equals 1 minus the refusal rate (19 of 177 answers were refusals) and shows that a source is attached, not that the model chose it correctly. It is a cheap proxy, not a human judgment.
- **Known misses:** 10 of 177 questions are not in the hybrid top 5 (q017, q045, q050, q099, q100, q121, q151, q152, q155, q158). The best-known: "pause the program for two seconds" retrieves M0/M1 *Program Pause* instead of G4 *Dwell*, because of the shared word "pause". A fix would be a small synonym map or a cross-encoder re-ranker. I did not tune retrieval against the test set.
- **A good RRF story for the q050 miss:** BM25 alone ranks G4 *first* for that question, but hybrid loses it. G4 is not
  in dense's top 50 chunks, so RRF gives it a single term (about 1/61). M60 and M0/M1 sit near the top of *both*
  lists and get two terms each (about 2/62). RRF rewards agreement between rankers, and that hurts when one ranker
  is confidently right and the other is wrong. A weighted fusion or a re-ranker would address it.
- **The corpus is LinuxCNC only.** Haas blocked the scraper.
- **The validator is not a simulator.** It knows nothing about fixtures, tool lengths or collisions, and it skips
  `#` parameters. The Safety sentence is in the README and the app footer.
- The 177 questions were written by me, not collected from real users, and the random dev/test split means the two halves differ by chance (test scored higher than dev).

## Demo

A five-minute script (start the app with `.venv/Scripts/python.exe app.py` and open http://127.0.0.1:7860):

1. **Ask the manual:** "Which words define the centre of an arc with G2 or G3?" Point at the cited section and its link
   to the exact anchor in the LinuxCNC docs. Then ask the G4 "pause for two seconds" question and show the failure
   honestly: it is the documented miss.
2. **Validate G-code:** the tab opens with a program that has no feed rate. Press Validate and show E005 on line 6.
   Then paste `examples/invalid_modal_conflict.nc` (shows E004 for `G0 G1` in one block). Switch the dialect to `haas`
   and paste `examples/valid_haas_face.nc`: zero findings.
3. **Generate G-code** (needs a key in `.env`): "Mill a 40 x 40 mm square pocket 3 mm deep with a 6 mm end mill". Show
   the attempts counter and the validation report. Without a key, show the disabled tab and explain the fallback.
4. **Close on the evaluation table** in the README and the dense-vs-BM25 breakdown. That is the engineering story:
   measure, explain why the numbers differ, and say what you would do next (a re-ranker, then a human-graded answer
   evaluation).
