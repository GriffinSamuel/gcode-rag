"""Tests for dense, BM25 and hybrid retrieval on the toy corpus (FakeEmbedder, offline)."""
from __future__ import annotations

import json

import numpy as np
import pytest

from gcoderag.embed import FakeEmbedder
from gcoderag.index import DenseIndex, rrf_fuse


@pytest.mark.parametrize("mode", ["dense", "bm25", "hybrid"])
@pytest.mark.parametrize("query,expected", [
    ("G4 dwell seconds", "linuxcnc:gcode:g4"),
    ("M3 spindle clockwise", "linuxcnc:mcode:m3-m4-m5"),
    ("tool change T word", "linuxcnc:mcode:m6"),
])
def test_obvious_chunk_ranks_first(toy_retriever, mode, query, expected) -> None:
    assert toy_retriever.retrieve(query, k=5, mode=mode)[0].doc_id == expected


def test_code_normalization_applies_to_queries(toy_retriever) -> None:
    assert toy_retriever.retrieve("G01 linear feed", k=3, mode="bm25")[0].doc_id == "linuxcnc:gcode:g1"
    assert toy_retriever.retrieve("G38.2 probe", k=3, mode="bm25")[0].doc_id == "linuxcnc:gcode:g38"


def test_rrf_math_hand_computed() -> None:
    scores = rrf_fuse([[1, 2, 3], [3, 1]], k=60)
    assert scores[1] == pytest.approx(1 / 61 + 1 / 62)
    assert scores[2] == pytest.approx(1 / 62)
    assert scores[3] == pytest.approx(1 / 63 + 1 / 61)
    assert sorted(scores, key=lambda r: -scores[r]) == [1, 3, 2]


def test_retrieve_docs_dedupes(toy_retriever) -> None:
    docs = toy_retriever.retrieve_docs("spindle", k=8, mode="hybrid")
    assert len(docs) == len(set(docs)) == 8


def test_hybrid_scores_are_rrf(toy_retriever) -> None:
    hits = toy_retriever.retrieve("rapid move", k=3, mode="hybrid")
    assert hits[0].doc_id == "linuxcnc:gcode:g0"
    assert hits[0].score <= 2 / 61 + 1e-12
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_save_load_roundtrip(tmp_path, toy_chunks) -> None:
    index = DenseIndex.build(toy_chunks, FakeEmbedder())
    index.save(tmp_path)
    meta = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
    assert meta["num_vectors"] == len(toy_chunks) == len(meta["chunk_ids"])
    loaded = DenseIndex.load(tmp_path, toy_chunks)
    assert np.array_equal(loaded.embeddings, index.embeddings)
    with pytest.raises(ValueError):
        DenseIndex.load(tmp_path, toy_chunks[:-1])
