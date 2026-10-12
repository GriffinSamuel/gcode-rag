---
title: CNC G-Code Assistant
emoji: 🛠️
colorFrom: gray
colorTo: blue
sdk: gradio
sdk_version: 6.30.0
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
---

# CNC G-Code Assistant

Ask questions about G-code and M-code and get answers from the LinuxCNC documentation, with a citation to the
exact manual section. Generate programs that a deterministic rule-based validator has checked (modal conflicts,
missing feed, envelope limits and more), or paste your own program into the validator.

- Retrieval: `BAAI/bge-small-en-v1.5` embeddings + BM25, fused with Reciprocal Rank Fusion (Recall@5 0.9464 and
  MRR 0.8363 on a 56-question evaluation set).
- On first start the Space downloads the LinuxCNC documentation and builds the index, which takes a few minutes.
- Without LLM secrets (`LLM_PROVIDER`, `LLM_API_KEY`, `LLM_MODEL`, `LLM_BASE_URL`) the Space runs in retrieval-only
  mode: it shows the cited passages, and generation is disabled. Validation always works.

Source code, evaluation and design decisions: see the GitHub repository README.

**Safety:** Generated G-code is for learning and must be checked in a simulator (such as NC Viewer) and reviewed by a qualified machinist before it is run on a real machine.
