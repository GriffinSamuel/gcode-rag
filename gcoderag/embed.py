"""Embedding backends: sentence-transformers, fastembed, and a deterministic fake (SPEC 4.1)."""
from __future__ import annotations

import hashlib
import logging
import os
from typing import Protocol, runtime_checkable

import numpy as np

from gcoderag.textnorm import tokenize

logger = logging.getLogger(__name__)

PRIMARY_MODEL = "BAAI/bge-small-en-v1.5"
FALLBACK_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 32
FAKE_DIM = 64


@runtime_checkable
class Embedder(Protocol):
    """Anything that turns texts into L2-normalized float32 row vectors."""

    name: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an array of shape (len(texts), dim), float32, L2-normalized rows."""
        ...


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Return a float32 copy of `matrix` with unit-length rows (zero rows stay zero)."""
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return (matrix / np.where(norms == 0, 1.0, norms)).astype(np.float32)


def default_model_name() -> str:
    """Model name from the EMBEDDING_MODEL environment variable, else the primary model."""
    return os.environ.get("EMBEDDING_MODEL") or PRIMARY_MODEL


class SentenceTransformerEmbedder:
    """Embedder backed by `sentence_transformers.SentenceTransformer` on CPU."""

    def __init__(self, model_name: str | None = None, batch_size: int = BATCH_SIZE) -> None:
        from sentence_transformers import SentenceTransformer

        self.name = model_name or default_model_name()
        self._batch_size = batch_size
        self._model = SentenceTransformer(self.name, device="cpu")
        get_dim = getattr(self._model, "get_embedding_dimension", None) or self._model.get_sentence_embedding_dimension
        self.dim = int(get_dim())

    def embed(self, texts: list[str]) -> np.ndarray:
        """Encode texts in batches with normalized embeddings."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = self._model.encode(texts, batch_size=self._batch_size, normalize_embeddings=True,
                                     show_progress_bar=False, convert_to_numpy=True)
        return _l2_normalize(vectors)


class FastEmbedEmbedder:
    """Embedder backed by `fastembed.TextEmbedding` (ONNX, no torch)."""

    def __init__(self, model_name: str | None = None, batch_size: int = BATCH_SIZE) -> None:
        from fastembed import TextEmbedding

        self.name = model_name or default_model_name()
        self._batch_size = batch_size
        self._model = TextEmbedding(model_name=self.name)
        self.dim = int(self.embed(["dimension probe"]).shape[1])

    def embed(self, texts: list[str]) -> np.ndarray:
        """Encode texts and L2-normalize the rows."""
        if not texts:
            return np.zeros((0, getattr(self, "dim", 0)), dtype=np.float32)
        return _l2_normalize(np.vstack(list(self._model.embed(texts, batch_size=self._batch_size))))


class FakeEmbedder:
    """Deterministic, offline bag-of-tokens embedder for tests: hash each token to a bucket, count, normalize."""

    def __init__(self, dim: int = FAKE_DIM) -> None:
        self.name = "fake-hash-embedder"
        self.dim = dim

    def _bucket(self, token: str) -> int:
        """Stable bucket index for a token (md5, independent of PYTHONHASHSEED)."""
        return int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % self.dim

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return token-count vectors, L2-normalized."""
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in tokenize(text):
                matrix[row, self._bucket(token)] += 1.0
        return _l2_normalize(matrix)


def get_embedder(name: str | None = None) -> Embedder:
    """Return a SentenceTransformer embedder, else FastEmbed; raise RuntimeError carrying both errors."""
    model_name = name or default_model_name()
    try:
        return SentenceTransformerEmbedder(model_name)
    except Exception as st_error:  # noqa: BLE001 - any import/download failure falls through
        logger.warning("sentence-transformers unavailable for %s: %s", model_name, st_error)
        try:
            return FastEmbedEmbedder(model_name)
        except Exception as fe_error:  # noqa: BLE001
            raise RuntimeError(
                f"no embedding backend for {model_name}: sentence-transformers: {st_error}; fastembed: {fe_error}"
            ) from fe_error
