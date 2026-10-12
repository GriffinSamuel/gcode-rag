"""Build the corpus files (SPEC 3.4): LinuxCNC (required) then Haas (best effort).

Shared by `scripts/build_corpus.py` and `app.py` (which rebuilds missing data on a fresh deploy).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from gcoderag.config import Paths
from gcoderag.ingest import haas, linuxcnc, write_jsonl
from gcoderag.ingest.chunker import chunk_docs

logger = logging.getLogger(__name__)


def build_corpus(paths: Paths, skip_haas: bool = False) -> dict[str, Any]:
    """Download, parse and chunk the sources; write docs.jsonl, chunks.jsonl and stats.json; return the stats."""
    linuxcnc_raw = paths.raw / "linuxcnc"
    linuxcnc.download_sources(linuxcnc_raw)
    lcnc_docs = linuxcnc.load_all(linuxcnc_raw)
    if skip_haas:
        logger.info("SKIP_HAAS set: Haas ingest skipped")
        haas_docs, haas_status = [], {"status": "skipped"}
    else:
        haas_docs, haas_status = haas.ingest(paths.raw / "haas")
    docs = lcnc_docs + haas_docs
    chunks = chunk_docs(docs)

    write_jsonl(paths.corpus / "docs.jsonl", docs)
    write_jsonl(paths.corpus / "chunks.jsonl", chunks)
    stats = {
        "num_docs": len(docs),
        "num_chunks": len(chunks),
        "sources": {"linuxcnc": len(lcnc_docs), "haas": len(haas_docs)},
        "distinct_codes": len({code for doc in docs for code in doc.codes}),
        "haas_status": haas_status["status"],
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (paths.corpus / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats
