# Final report: gcode-rag

Read this before an interview. It tells you what the project is, which numbers to quote, what to admit, and what to
demo in five minutes.

## What was built

**One-sentence pitch:** "A RAG assistant for CNC machinists. It answers G-code questions with citations to the exact
section of the LinuxCNC manual. It generates G-code programs and checks them with a deterministic validator, and
when the validator finds errors it sends them back to the model for another attempt. I measured the retrieval on a
56-question test set."

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
7. **Gradio app** with three tabs. It still works without an API key (retrieval-only).
8. **Tests:** 95 offline tests that run in about 10 seconds, using a fake embedder and a scripted fake LLM.

## Results

Quote these (from `results/results.json`, 56 questions, 44 of them phrased without naming the code):

| config | recall@1 | recall@5 | MRR |
| --- | --- | --- | --- |
| dense_baseline | 0.5714 | 0.9464 | 0.6961 |
| bm25_only | 0.5714 | 0.8571 | 0.7008 |
| hybrid_rrf | 0.7500 | 0.9464 | 0.8363 |

- **The line to remember:** "Hybrid retrieval finds a correct section in the top 5 for 95 % of questions, and
  ranks it first for 75 %, up from 57 % with either method alone."
- **Why hybrid wins (this is the interesting part):** on the 12 questions that name a code, BM25 ranks the right
  section first 92 % of the time and dense only 58 %, because embeddings blur `G1`/`G10`/`G41`. On the 44
  paraphrased questions it flips: dense reaches 0.93 Recall@5 and BM25 0.82. RRF keeps the strengths of both.
- **Validator:** all 7 example programs are classified correctly. The 3 valid ones have zero findings, and each of
  the 4 invalid ones triggers exactly the intended rules.
- **Integrity point:** the gold answers were written before the first evaluation run and never changed afterwards,
  and no retrieval tuning was done against the test set.

## Limitations

Raise these yourself before you are asked:

- **Answer accuracy is not measured yet.** The build ran without an LLM key, so the keyword-accuracy evaluation is
  recorded as `skipped_no_llm_key`. The code path is built and tested with a fake LLM. To measure it, add a key to
  `.env` and run `scripts/run_eval.py`. Keyword accuracy is a cheap proxy, not a human judgment.
- **Known misses:** "pause the program for two seconds" retrieves M0/M1 *Program Pause* instead of G4 *Dwell*, because
  of the shared word "pause". G53 (machine coordinates) lands at rank 7, and the override controls M48/M49 at rank 8.
  A fix would be a small synonym map or a cross-encoder re-ranker. I did not tune against the test set.
- **A good RRF story for the q050 miss:** BM25 alone ranks G4 *first* for that question, but hybrid loses it. G4 is not
  in dense's top 50 chunks, so RRF gives it a single term (about 1/61). M60 and M0/M1 sit near the top of *both*
  lists and get two terms each (about 2/62). RRF rewards agreement between rankers, and that hurts when one ranker
  is confidently right and the other is wrong. A weighted fusion or a re-ranker would address it.
- **The corpus is LinuxCNC only.** Haas blocked the scraper.
- **The validator is not a simulator.** It knows nothing about fixtures, tool lengths or collisions, and it skips
  `#` parameters. The Safety sentence is in the README and the app footer.
- The 56 questions were written by me, not collected from real users.

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
