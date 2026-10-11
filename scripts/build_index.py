#!/usr/bin/env python
"""Embed data/corpus/chunks.jsonl and write data/index/embeddings.npy + meta.json (SPEC 4.5)."""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gcoderag.config import get_paths, load_settings  # noqa: E402
from gcoderag.embed import get_embedder  # noqa: E402
from gcoderag.index import DenseIndex  # noqa: E402
from gcoderag.ingest import load_chunks  # noqa: E402


def main() -> int:
    """Build and save the dense index; print model, dim, num_vectors and seconds."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    load_settings()
    paths = get_paths()
    chunks = load_chunks(paths.corpus / "chunks.jsonl")
    start = time.perf_counter()
    embedder = get_embedder()
    index = DenseIndex.build(chunks, embedder)
    index.save(paths.index)
    seconds = time.perf_counter() - start
    print(f"model: {index.embedding_model}\ndim: {index.dim}\nnum_vectors: {len(index.chunk_ids)}\nseconds: {seconds:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
