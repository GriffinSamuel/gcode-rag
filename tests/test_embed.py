"""Tests for the embedders; only FakeEmbedder runs offline, the real model is a slow test."""
from __future__ import annotations

import os

import numpy as np
import pytest

from gcoderag.embed import Embedder, FakeEmbedder


def test_fake_embedder_shape_and_normalization() -> None:
    vectors = FakeEmbedder().embed(["G1 linear move", "spindle clockwise", ""])
    assert vectors.shape == (3, 64) and vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(vectors[:2], axis=1), 1.0)
    assert np.allclose(vectors[2], 0.0)


def test_fake_embedder_is_deterministic() -> None:
    a = FakeEmbedder().embed(["rapid move G0"])
    b = FakeEmbedder().embed(["rapid move G0"])
    assert np.array_equal(a, b)
    assert isinstance(FakeEmbedder(), Embedder)


@pytest.mark.slow
@pytest.mark.skipif(not os.getenv("RUN_SLOW"), reason="needs the real embedding model")
def test_real_embedder_similarity() -> None:
    from gcoderag.embed import get_embedder

    embedder = get_embedder()
    vecs = embedder.embed(["pause the program for two seconds", "G4 Dwell: axes remain unmoving", "spindle coolant"])
    assert vecs.shape[1] == embedder.dim
    assert float(vecs[0] @ vecs[1]) > float(vecs[0] @ vecs[2])
