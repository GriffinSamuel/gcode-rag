"""Shared fixtures: the 8-chunk toy corpus and an offline retriever over it."""
from __future__ import annotations

from pathlib import Path

import pytest

from gcoderag.embed import FakeEmbedder
from gcoderag.index import HybridRetriever, build_retriever
from gcoderag.ingest import Chunk, load_chunks

TOY_CORPUS = Path(__file__).parent / "fixtures" / "toy_corpus.jsonl"


@pytest.fixture(scope="session")
def toy_chunks() -> list[Chunk]:
    return load_chunks(TOY_CORPUS)


@pytest.fixture(scope="session")
def toy_retriever(toy_chunks: list[Chunk]) -> HybridRetriever:
    return build_retriever(toy_chunks, FakeEmbedder())
