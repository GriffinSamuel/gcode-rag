"""Corpus ingestion: LinuxCNC AsciiDoc sources, Haas pages (best effort), chunking.

Also defines the `Doc` and `Chunk` records shared by ingestion, indexing and evaluation.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

USER_AGENT = "gcode-rag/0.1 (student portfolio project)"
HTTP_TIMEOUT_S = 30
HTTP_RETRIES = 2


@dataclass(frozen=True)
class Doc:
    """One documentation section (SPEC 3.3)."""

    doc_id: str
    source: str
    title: str
    codes: list[str]
    url: str
    section_path: str
    text: str


@dataclass(frozen=True)
class Chunk:
    """A retrievable slice of a document (SPEC 3.4)."""

    chunk_id: str
    doc_id: str
    source: str
    title: str
    codes: list[str] = field(default_factory=list)
    url: str = ""
    chunk_index: int = 0
    text: str = ""


def slugify(text: str, max_len: int = 60) -> str:
    """Lowercase, replace runs of non-alphanumerics with '-', trim dashes, cut to `max_len`."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_len].rstrip("-") or "section"


def write_jsonl(path: Path, records: Iterable[Any]) -> int:
    """Write dataclass records (or dicts) as UTF-8 JSON lines; return the number written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for record in records:
            row = asdict(record) if hasattr(record, "__dataclass_fields__") else record
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a UTF-8 JSON-lines file into a list of dicts, skipping blank lines."""
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_docs(path: Path) -> list[Doc]:
    """Load `docs.jsonl` into `Doc` records."""
    return [Doc(**row) for row in read_jsonl(path)]


def load_chunks(path: Path) -> list[Chunk]:
    """Load `chunks.jsonl` into `Chunk` records."""
    return [Chunk(**row) for row in read_jsonl(path)]
