"""Library of validator-clean example programs, retrieved as few-shot references for the generator.

Each file in `examples/library/` starts with a `(dialect: ...)` comment and a `(request: ...)` comment describing the
request the program answers. The request descriptions are embedded with the project's embedder; `similar()` returns the
programs whose descriptions are closest to a new request.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gcoderag.embed import Embedder
from gcoderag.textnorm import normalize_codes

logger = logging.getLogger(__name__)

EXAMPLES_PER_REQUEST_DEFAULT = 2   # reference programs shown to the model per request

_REQUEST_RE = re.compile(r"\(\s*request\s*:\s*(.*?)\)\s*$", re.IGNORECASE | re.MULTILINE)
_DIALECT_RE = re.compile(r"\(\s*dialect\s*:\s*(\w+)\s*\)", re.IGNORECASE)


@dataclass(frozen=True)
class Example:
    """One library program: its file stem, dialect, the request it answers, and the full program text."""

    name: str
    dialect: str
    request: str
    program: str


def parse_example(name: str, text: str) -> Example:
    """Build an Example from file text; ValueError if the `(request: ...)` or `(dialect: ...)` header is missing."""
    request = _REQUEST_RE.search(text)
    dialect = _DIALECT_RE.search(text)
    if not request or not dialect:
        raise ValueError(f"{name}: needs '(dialect: ...)' and '(request: ...)' header comments")
    return Example(name, dialect.group(1).lower(), request.group(1).strip(), text.strip())


def default_library_dir() -> Path:
    """`examples/library/` in the repository."""
    from gcoderag.config import get_paths

    return get_paths().root / "examples" / "library"


def load_examples(directory: Path | None = None) -> list[Example]:
    """Load every `*.nc` file in the library directory, sorted by name."""
    directory = directory or default_library_dir()
    return [parse_example(p.stem, p.read_text(encoding="utf-8")) for p in sorted(directory.glob("*.nc"))]


class ExampleLibrary:
    """Embeds the example request descriptions once and returns the closest examples for a new request."""

    def __init__(self, examples: list[Example], embedder: Embedder) -> None:
        if not examples:
            raise ValueError("an example library needs at least one example")
        self.examples = list(examples)
        self.embedder = embedder
        self._vectors = embedder.embed([normalize_codes(e.request) for e in self.examples])

    @classmethod
    def from_directory(cls, embedder: Embedder, directory: Path | None = None) -> ExampleLibrary:
        """Load the library from disk (default: examples/library/) and embed it."""
        return cls(load_examples(directory), embedder)

    def similar(self, request: str, k: int = 2, dialect: str | None = None) -> list[Example]:
        """The `k` examples whose request descriptions are most similar to `request`, best first.

        With `dialect`, only examples of that dialect are considered; if fewer than `k` exist, the best of the
        other dialect fill the remaining slots (after the same-dialect ones).
        """
        query = self.embedder.embed([normalize_codes(request)])[0]
        scores = self._vectors @ query
        order = [int(i) for i in np.argsort(-scores, kind="stable")]
        if dialect is not None:
            same = [i for i in order if self.examples[i].dialect == dialect]
            order = same + [i for i in order if i not in same]
        return [self.examples[i] for i in order[:k]]
