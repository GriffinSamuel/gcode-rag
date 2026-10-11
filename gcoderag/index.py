"""Dense, BM25 and hybrid RRF retrieval over chunks (SPEC 4.2, 4.4)."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
from rank_bm25 import BM25Okapi

from gcoderag.embed import Embedder
from gcoderag.ingest import Chunk
from gcoderag.textnorm import normalize_codes, tokenize

logger = logging.getLogger(__name__)

RRF_K = 60
FUSION_DEPTH = 50
Mode = Literal["dense", "bm25", "hybrid"]
MODES: tuple[Mode, ...] = ("dense", "bm25", "hybrid")


@dataclass
class Hit:
    """One retrieved chunk with its score under the chosen mode."""

    chunk_id: str
    doc_id: str
    score: float
    text: str
    title: str
    url: str
    codes: list[str] = field(default_factory=list)
    source: str = ""


def index_text(chunk: Chunk) -> str:
    """Text that is embedded / BM25-indexed: code-normalized, with the title prepended to non-first chunks."""
    body = chunk.text if chunk.chunk_index == 0 else f"{chunk.title}\n\n{chunk.text}"
    return normalize_codes(body)


def _top_k(scores: np.ndarray, k: int) -> list[tuple[int, float]]:
    """Return (row, score) for the k highest scores, sorted desc; ties keep the lower row first."""
    order = np.argsort(-scores, kind="stable")[:k]
    return [(int(i), float(scores[i])) for i in order]


class DenseIndex:
    """Exact cosine search over L2-normalized chunk embeddings."""

    def __init__(self, embeddings: np.ndarray, chunk_ids: list[str], embedding_model: str) -> None:
        if embeddings.shape[0] != len(chunk_ids):
            raise ValueError(f"{embeddings.shape[0]} vectors but {len(chunk_ids)} chunk ids")
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.chunk_ids = list(chunk_ids)
        self.embedding_model = embedding_model

    @classmethod
    def build(cls, chunks: list[Chunk], embedder: Embedder) -> "DenseIndex":
        """Embed every chunk's index text."""
        vectors = embedder.embed([index_text(c) for c in chunks])
        return cls(vectors, [c.chunk_id for c in chunks], embedder.name)

    @property
    def dim(self) -> int:
        """Embedding dimension."""
        return int(self.embeddings.shape[1]) if self.embeddings.ndim == 2 else 0

    def save(self, index_dir: Path) -> None:
        """Write embeddings.npy (float32) and meta.json."""
        index_dir.mkdir(parents=True, exist_ok=True)
        np.save(index_dir / "embeddings.npy", self.embeddings)
        meta = {
            "embedding_model": self.embedding_model,
            "dim": self.dim,
            "num_vectors": len(self.chunk_ids),
            "chunk_ids": self.chunk_ids,
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        (index_dir / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, index_dir: Path, chunks: list[Chunk]) -> "DenseIndex":
        """Load a saved index and check it matches `chunks` (same ids, same order)."""
        meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
        expected = [c.chunk_id for c in chunks]
        if meta["chunk_ids"] != expected:
            raise ValueError("index chunk_ids do not match chunks.jsonl; rebuild with scripts/build_index.py")
        return cls(np.load(index_dir / "embeddings.npy"), meta["chunk_ids"], meta["embedding_model"])

    def search(self, query_vec: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Return (row, cosine) pairs sorted by descending similarity."""
        return _top_k(self.embeddings @ np.asarray(query_vec, dtype=np.float32).reshape(-1), k)


class BM25Index:
    """Okapi BM25 over code-normalized tokens."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self._bm25 = BM25Okapi([tokenize(index_text(c)) or ["_empty_"] for c in chunks])

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        """Return (row, bm25 score) pairs sorted by descending score."""
        tokens = tokenize(normalize_codes(query))
        if not tokens:
            return []
        return _top_k(np.asarray(self._bm25.get_scores(tokens), dtype=np.float64), k)


def rrf_fuse(rankings: list[list[int]], k: int = RRF_K) -> dict[int, float]:
    """Reciprocal rank fusion: score(d) = sum over rankings of 1 / (k + rank), ranks starting at 1."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, row in enumerate(ranking, start=1):
            scores[row] = scores.get(row, 0.0) + 1.0 / (k + rank)
    return scores


class HybridRetriever:
    """Dense, BM25 or RRF-fused retrieval returning `Hit`s."""

    def __init__(self, chunks: list[Chunk], dense: DenseIndex, bm25: BM25Index, embedder: Embedder) -> None:
        self.chunks = chunks
        self.dense = dense
        self.bm25 = bm25
        self.embedder = embedder

    def _dense_search(self, query: str, k: int) -> list[tuple[int, float]]:
        return self.dense.search(self.embedder.embed([normalize_codes(query)])[0], k)

    def _ranked_rows(self, query: str, k: int, mode: Mode) -> list[tuple[int, float]]:
        """Rows and scores for a mode, best first."""
        if mode == "dense":
            return self._dense_search(query, k)
        if mode == "bm25":
            return self.bm25.search(query, k)
        if mode != "hybrid":
            raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
        dense_hits = self._dense_search(query, FUSION_DEPTH)
        bm25_hits = self.bm25.search(query, FUSION_DEPTH)
        fused = rrf_fuse([[row for row, _ in dense_hits], [row for row, _ in bm25_hits]])
        dense_score = dict(dense_hits)
        order = sorted(fused, key=lambda row: (-fused[row], -dense_score.get(row, float("-inf")), row))
        return [(row, fused[row]) for row in order[:k]]

    def retrieve(self, query: str, k: int = 10, mode: Mode = "hybrid") -> list[Hit]:
        """Return the top `k` chunks for `query` under `mode`."""
        hits: list[Hit] = []
        for row, score in self._ranked_rows(query, k, mode):
            chunk = self.chunks[row]
            hits.append(Hit(chunk_id=chunk.chunk_id, doc_id=chunk.doc_id, score=score, text=chunk.text,
                            title=chunk.title, url=chunk.url, codes=list(chunk.codes), source=chunk.source))
        return hits

    def retrieve_docs(self, query: str, k: int = 10, mode: Mode = "hybrid") -> list[str]:
        """Distinct doc ids in rank order from the top `k` chunks."""
        return list(dict.fromkeys(hit.doc_id for hit in self.retrieve(query, k, mode)))


def build_retriever(chunks: list[Chunk], embedder: Embedder, index_dir: Path | None = None) -> HybridRetriever:
    """Create a retriever; load the dense index from `index_dir` when it matches, else build (and save) it."""
    dense: DenseIndex | None = None
    if index_dir is not None and (index_dir / "meta.json").exists():
        try:
            dense = DenseIndex.load(index_dir, chunks)
            if dense.embedding_model != embedder.name:
                logger.warning("index built with %s but embedder is %s; rebuilding", dense.embedding_model, embedder.name)
                dense = None
        except (ValueError, OSError, KeyError) as exc:
            logger.warning("could not load index (%s); rebuilding", exc)
            dense = None
    if dense is None:
        dense = DenseIndex.build(chunks, embedder)
        if index_dir is not None:
            dense.save(index_dir)
    return HybridRetriever(chunks, dense, BM25Index(chunks), embedder)
