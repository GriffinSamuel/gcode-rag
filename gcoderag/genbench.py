"""Generation benchmark: run the generator over a fixed set of requests and summarise how often it gets it right."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from gcoderag.ingest import read_jsonl
from gcoderag.validator import AXIS_CONSUMERS, FEED_MOTION, MODAL_GROUPS, code_name, parse_program

logger = logging.getLogger(__name__)

DECIMALS = 4
RUN_LABELS = ("without_examples", "with_examples")


class Generates(Protocol):
    """The part of GcodeGenerator the benchmark needs."""

    def generate(self, request: str, dialect: Any = ..., max_attempts: int = ...) -> Any:
        """Return a GenResult."""
        ...


def load_requests(path: Path) -> list[dict[str, Any]]:
    """Load `eval/gen_requests.jsonl` (id, dialect, request)."""
    rows = read_jsonl(path)
    for row in rows:
        if not {"id", "dialect", "request"} <= set(row):
            raise ValueError(f"bad request row: {row}")
    return rows


def has_feed_move_below_zero(program: str) -> bool:
    """True if some feed move (G1/G2/G3 or a canned cycle) ends with Z below 0 (G91 moves are accumulated)."""
    z = 0.0
    incremental = False
    motion: str | None = None
    for block in parse_program(program):
        if block.skipped:
            continue
        codes = [code_name(w.letter, w.value) for w in block.words if w.letter in "GM"]
        if "G91" in codes:
            incremental = True
        elif "G90" in codes:
            incremental = False
        motions = [c for c in codes if c in MODAL_GROUPS["G motion"]]
        if motions:
            motion = ([c for c in motions if c != "G80"] or motions)[-1]
        if any(c in AXIS_CONSUMERS for c in codes):
            continue
        for w in block.words:
            if w.letter == "Z":
                z = z + w.value if incremental else w.value
                if motion in FEED_MOTION and z < 0:
                    return True
    return False


def run_request(generator: Generates, item: dict[str, Any]) -> dict[str, Any]:
    """Generate one program and record the benchmark columns (an LLM failure counts as a failed request)."""
    try:
        result = generator.generate(item["request"], dialect=item["dialect"])
    except Exception as exc:  # noqa: BLE001 - provider errors are recorded, not fatal for the whole benchmark
        logger.error("request %s failed: %s", item["id"], exc)
        return {"id": item["id"], "dialect": item["dialect"], "request": item["request"], "first_attempt_ok": False,
                "final_ok": False, "attempts": 0, "warnings_final": None, "feed_below_z0": False,
                "errors_final": [], "error": f"{type(exc).__name__}: {exc}", "program": ""}
    return {
        "id": item["id"], "dialect": item["dialect"], "request": item["request"],
        "first_attempt_ok": result.history[0]["n_errors"] == 0,
        "final_ok": result.report.ok,
        "attempts": result.attempts,
        "warnings_final": len(result.report.warnings),
        "feed_below_z0": has_feed_move_below_zero(result.program),
        "errors_final": [f"{i.rule} line {i.line_no}: {i.message}" for i in result.report.errors],
        "program": result.program,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Rates over a run: first-attempt pass, final pass, mean attempts, mean warnings, clean and cutting shares."""
    n = len(rows)
    if not n:
        raise ValueError("no benchmark rows")
    answered = [r for r in rows if r["warnings_final"] is not None]
    return {
        "n": n,
        "first_attempt_ok_rate": round(sum(r["first_attempt_ok"] for r in rows) / n, DECIMALS),
        "final_ok_rate": round(sum(r["final_ok"] for r in rows) / n, DECIMALS),
        "mean_attempts": round(sum(r["attempts"] for r in rows) / n, DECIMALS),
        "mean_warnings_final": round(sum(r["warnings_final"] for r in answered) / max(len(answered), 1), DECIMALS),
        "zero_warning_rate": round(sum(r["final_ok"] and r["warnings_final"] == 0 for r in rows) / n, DECIMALS),
        "feed_below_z0_rate": round(sum(r["feed_below_z0"] for r in rows) / n, DECIMALS),
    }


def update_results_file(path: Path, label: str, run: dict[str, Any]) -> dict[str, Any]:
    """Store `run` under `runs[label]` in the results JSON (keeping the other run) and return the whole document."""
    if label not in RUN_LABELS:
        raise ValueError(f"label must be one of {RUN_LABELS}")
    doc: dict[str, Any] = {"runs": {}}
    if path.exists():
        doc = json.loads(path.read_text(encoding="utf-8"))
        doc.setdefault("runs", {})
    doc["runs"][label] = run
    doc["runs"] = {k: doc["runs"][k] for k in RUN_LABELS if k in doc["runs"]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(doc, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    return doc


def make_run(label: str, model: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Assemble one run record: metadata, summary rates, and the per-request list."""
    return {"label": label, "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "model": model,
            "examples": label == "with_examples", **summarize(rows), "requests": rows}
