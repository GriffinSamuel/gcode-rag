#!/usr/bin/env python
"""Validate every examples/*.nc file with the dialect named in its first comment (SPEC 7.5).

Writes results/validator_report.json; `expected` comes from the filename prefix (valid_ / invalid_).
Exits 1 if any example does not match its expectation.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gcoderag.config import get_paths  # noqa: E402
from gcoderag.validator import GcodeValidator, detect_dialect  # noqa: E402


def check_example(path: Path, root: Path) -> dict:
    """Validate one example file and compare the outcome with its filename prefix."""
    text = path.read_text(encoding="utf-8")
    dialect = detect_dialect(text) or "linuxcnc"
    expected = "invalid" if path.name.startswith("invalid_") else "valid"
    report = GcodeValidator(dialect).validate(text)
    return {
        "file": path.relative_to(root).as_posix(),
        "dialect": dialect,
        "expected": expected,
        "ok": report.ok,
        "errors": len(report.errors),
        "warnings": len(report.warnings),
        "rules": sorted({i.rule for i in report.errors + report.warnings}),
        "matches_expectation": report.ok == (expected == "valid"),
    }


def main() -> int:
    """Validate all examples, write the report, print one line per file."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    paths = get_paths()
    files = sorted((paths.root / "examples").glob("*.nc"))
    if not files:
        print("no examples/*.nc files found", file=sys.stderr)
        return 1
    results = [check_example(p, paths.root) for p in files]
    report = {"generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "results": results}
    out = paths.results / "validator_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for r in results:
        mark = "OK  " if r["matches_expectation"] else "FAIL"
        print(f"{mark} {r['file']:<42} {r['dialect']:<9} expected={r['expected']:<7} ok={r['ok']!s:<5} "
              f"rules={','.join(r['rules']) or '-'}")
    print(f"wrote {out}")
    return 0 if all(r["matches_expectation"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
