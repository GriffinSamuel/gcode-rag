#!/usr/bin/env python
"""Build the corpus: LinuxCNC (required) then Haas (best effort) -> data/corpus/{docs,chunks}.jsonl + stats.json.

Set SKIP_HAAS=1 to skip the Haas scrape.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gcoderag.config import get_paths  # noqa: E402
from gcoderag.ingest.build import build_corpus  # noqa: E402


def main() -> int:
    """Download, parse, chunk and write the corpus; print the stats."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    stats = build_corpus(get_paths(), skip_haas=os.getenv("SKIP_HAAS") == "1")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
