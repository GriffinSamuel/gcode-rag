"""Smoke test: the package imports and exposes its version."""
from __future__ import annotations

import gcoderag
from gcoderag.config import get_paths


def test_package_imports() -> None:
    assert gcoderag.__version__ == "0.1.0"
    assert get_paths().corpus.name == "corpus"
