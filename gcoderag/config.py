"""Repository paths and environment settings."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

SETTING_PREFIXES = ("LLM_", "HF_")
SETTING_KEYS = ("EMBEDDING_MODEL", "ANTHROPIC_API_KEY")


@dataclass(frozen=True)
class Paths:
    """Well-known directories of the repository."""

    root: Path
    data: Path
    raw: Path
    corpus: Path
    index: Path
    results: Path
    eval: Path


def get_paths(root: Path | None = None) -> Paths:
    """Return the repository paths; `root` defaults to the parent of the `gcoderag/` package."""
    base = (root or Path(__file__).resolve().parent.parent).resolve()
    data = base / "data"
    return Paths(
        root=base,
        data=data,
        raw=data / "raw",
        corpus=data / "corpus",
        index=data / "index",
        results=base / "results",
        eval=base / "eval",
    )


def load_settings() -> dict[str, str]:
    """Load `.env` (if present) into the environment and return the LLM_*, HF_* and EMBEDDING_MODEL values."""
    try:
        from dotenv import load_dotenv

        load_dotenv(get_paths().root / ".env", override=False)
    except ImportError:  # python-dotenv is optional at runtime
        pass
    return {
        key: value
        for key, value in os.environ.items()
        if key.startswith(SETTING_PREFIXES) or key in SETTING_KEYS
    }
