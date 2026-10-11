# Decision log

Append one line per decision. Format: `- YYYY-MM-DD HH:MM | <topic> | <decision> | <why>`.
Also log: escalations to `fable-escalation` (brief + outcome), eval iterations (before/after numbers), fallbacks taken.

- 2026-10-11 13:00 | planning | Primary corpus is LinuxCNC AsciiDoc from the GitHub repo; Haas pages are best effort and local only | LinuxCNC source is open and robots-friendly; linuxcnc.org itself disallows crawlers; Haas pages are copyrighted and may be JS-rendered
- 2026-10-11 13:00 | planning | Hand-written dense + BM25 + RRF retriever (numpy, rank_bm25) instead of a vector DB | Corpus is a few hundred chunks; exact cosine search is instant; shows understanding of the retrieval math and avoids Windows install risk
- 2026-10-11 13:00 | planning | Embeddings via sentence-transformers BAAI/bge-small-en-v1.5, fastembed fallback | Small, CPU-fast, strong on short technical text
- 2026-10-11 13:00 | planning | LLM is pluggable (Anthropic or any OpenAI-compatible endpoint); project must fully build and test without a key | Unattended build cannot depend on a paid key; tests use FakeLLM
- 2026-10-11 13:00 | planning | Validator is deterministic rule-based Python, not an LLM | Safety checks must be reproducible and explainable in interviews
- 2026-10-11 13:00 | planning | Definition of done is scripts/verify_done.py, enforced by a Stop hook and /goal | Prevents declaring victory early; makes the one-shot run converge
- 2026-10-11 14:40 | environment | Python 3.11.15 in .venv; embedding backend sentence-transformers 6.1.0 on torch 2.14.1+cpu (fastembed fallback not needed) | requirements.txt installed cleanly; import sentence_transformers works
