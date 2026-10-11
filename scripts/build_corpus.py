#!/usr/bin/env python
"""Build the corpus: LinuxCNC (required) then Haas (best effort) -> data/corpus/{docs,chunks}.jsonl + stats.json."""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gcoderag.config import get_paths  # noqa: E402
from gcoderag.ingest import haas, linuxcnc, write_jsonl  # noqa: E402
from gcoderag.ingest.chunker import chunk_docs  # noqa: E402


def main() -> int:
    """Download, parse, chunk and write the corpus; print the stats."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    paths = get_paths()
    linuxcnc_raw = paths.raw / "linuxcnc"
    linuxcnc.download_sources(linuxcnc_raw)
    lcnc_docs = linuxcnc.load_all(linuxcnc_raw)
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
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
