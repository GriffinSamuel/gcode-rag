"""Claude Code Stop hook: refuse to let the session end until scripts/verify_done.py passes.

How it works (see https://code.claude.com/docs/en/hooks):
- Claude Code runs this every time Claude tries to finish a turn, piping event JSON to stdin.
- We run the project's definition-of-done script. If it passes, we write a DONE marker and allow the stop.
- Otherwise we print {"decision": "block", "reason": ...}. The reason is fed back to Claude as its next
  instruction, so it contains the failing checks and the unchecked PROGRESS items: a self-directing loop.
- Stuck detection: if Claude keeps stopping without PROGRESS.md or the git HEAD changing, the reason switches
  to an escalation order (fable-escalation subagent). After STUCK_ALLOW_STOP consecutive no-progress stops we
  allow the stop so the run cannot burn budget forever.
- We deliberately do NOT exit early on `stop_hook_active`: Claude makes tool calls between stops, which resets
  Claude Code's consecutive-block cap, so the loop keeps going as long as work is happening.

Standard library only. Logs to .claude/hooks/state/stop_gate.log. Output is ASCII-only JSON on stdout.
To re-arm after completion, delete .claude/hooks/state/DONE.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ESCALATE_AFTER = 4          # consecutive stops without progress before demanding escalation
STUCK_ALLOW_STOP = 10       # consecutive stops without progress before we give up and allow the stop
MAX_TOTAL_BLOCKS = 300      # absolute cap on continuations for one project
VERIFY_TIMEOUT_S = 840      # verify_done runs pytest; the hook timeout in settings.json is 900


def _root() -> Path:
    env_root = os.environ.get("CLAUDE_PROJECT_DIR")
    return Path(env_root).resolve() if env_root else Path.cwd().resolve()


def _read_int(path: Path, default: int = 0) -> int:
    try:
        return int(path.read_text(encoding="utf-8").strip() or default)
    except (OSError, ValueError):
        return default


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _fingerprint(root: Path) -> str:
    h = hashlib.sha1()
    progress = root / "PROGRESS.md"
    if progress.exists():
        h.update(progress.read_bytes())
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(root), capture_output=True, text=True, timeout=20)
        h.update((head.stdout or "").encode("utf-8"))
    except (OSError, subprocess.TimeoutExpired):
        pass
    return h.hexdigest()


def _unchecked(root: Path, limit: int = 12) -> list[str]:
    progress = root / "PROGRESS.md"
    if not progress.exists():
        return ["PROGRESS.md is missing"]
    items = [ln.strip() for ln in progress.read_text(encoding="utf-8", errors="replace").splitlines()
             if ln.strip().startswith("- [ ]")]
    return items[:limit]


def _python_for_verify(root: Path) -> str:
    for cand in (root / ".venv" / "Scripts" / "python.exe", root / ".venv" / "bin" / "python"):
        if cand.exists():
            return str(cand)
    return sys.executable


def _run_verify(root: Path) -> tuple[bool, str]:
    script = root / "scripts" / "verify_done.py"
    if not script.exists():
        return False, "scripts/verify_done.py is missing; restore it from the scaffold (it is the definition of done)."
    try:
        proc = subprocess.run([_python_for_verify(root), str(script)], cwd=str(root), capture_output=True, text=True,
                              timeout=VERIFY_TIMEOUT_S, encoding="utf-8", errors="replace")
        out = (proc.stdout or "") + (proc.stderr or "")
        passed = proc.returncode == 0 and "ALL CHECKS PASSED" in out
        return passed, out
    except subprocess.TimeoutExpired:
        return False, f"verify_done.py timed out after {VERIFY_TIMEOUT_S}s (tests too slow? mark network/model tests slow)."
    except OSError as exc:
        return False, f"could not run verify_done.py: {exc}"


def _tail(text: str, max_lines: int = 45, max_chars: int = 3500) -> str:
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    keep = [ln for ln in lines if ln.startswith(("FAIL", "CHECKS FAILED", " - ")) or "passed" in ln or "failed" in ln]
    chosen = keep if keep else lines[-max_lines:]
    joined = "\n".join(chosen[-max_lines:])
    return joined[-max_chars:]


def main() -> int:
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except json.JSONDecodeError:
        payload = {}

    root = _root()
    state = root / ".claude" / "hooks" / "state"
    state.mkdir(parents=True, exist_ok=True)
    log = state / "stop_gate.log"
    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def logline(msg: str) -> None:
        with log.open("a", encoding="utf-8") as fh:
            fh.write(f"{now} {msg}\n")

    if (state / "DONE").exists():
        logline("DONE marker present; allowing stop")
        return 0

    count = _read_int(state / "count") + 1
    _write(state / "count", str(count))

    fp = _fingerprint(root)
    prev_fp = (state / "fingerprint").read_text(encoding="utf-8").strip() if (state / "fingerprint").exists() else ""
    streak = _read_int(state / "streak")
    streak = streak + 1 if fp == prev_fp else 0
    _write(state / "fingerprint", fp)
    _write(state / "streak", str(streak))

    if count > MAX_TOTAL_BLOCKS:
        logline(f"cap reached ({count}); allowing stop")
        print(json.dumps({"systemMessage": f"Stop gate: reached the cap of {MAX_TOTAL_BLOCKS} continuations; allowing stop."}, ensure_ascii=True))
        return 0
    if streak >= STUCK_ALLOW_STOP:
        logline(f"no progress for {streak} stops; allowing stop")
        print(json.dumps({"systemMessage": (f"Stop gate: no progress for {streak} consecutive stops. Allowing stop. "
                                            "Read .claude/hooks/state/stop_gate.log and docs/DECISIONS.md, then restart with /goal.")},
                         ensure_ascii=True))
        return 0

    passed, output = _run_verify(root)
    if passed:
        _write(state / "DONE", now)
        logline(f"verify_done PASSED on stop #{count}; DONE marker written")
        print(json.dumps({"systemMessage": "Stop gate: verify_done.py passed. Project complete. DONE marker written."}, ensure_ascii=True))
        return 0

    unchecked = _unchecked(root)
    parts: list[str] = []
    if streak >= ESCALATE_AFTER:
        parts.append(
            f"ESCALATION REQUIRED: you have stopped {streak} times in a row without PROGRESS.md or git HEAD changing. "
            "Delegate the current blocker to the `fable-escalation` subagent NOW with a complete brief (PROGRESS item, "
            "exact commands and full error text, what you tried, file paths, the acceptance check). Apply its fix, log "
            "the outcome in docs/DECISIONS.md, commit, and continue. If the blocker is an external dependency, apply the "
            "fallback ladder in CLAUDE.md section 5 instead."
        )
    parts.append(
        "STOP GATE: the project is NOT finished. Do not stop and do not ask the user anything. Continue working now. "
        "Rules: work through PROGRESS.md in order, tick boxes as items pass, commit after each item, never leave the tree "
        "dirty, and re-run `.venv/Scripts/python.exe scripts/verify_done.py` before trying to stop again."
    )
    parts.append("verify_done.py failures:\n" + _tail(output))
    if unchecked:
        parts.append("Unchecked PROGRESS.md items (do the first one next):\n" + "\n".join(unchecked))
    reason = "\n\n".join(parts)
    logline(f"blocked stop #{count} (streak {streak}); unchecked={len(unchecked)}")
    print(json.dumps({"decision": "block", "reason": reason}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001  never crash the hook; allow the stop with a note
        print(json.dumps({"systemMessage": f"Stop gate error (allowing stop): {exc!r}"}, ensure_ascii=True))
        sys.exit(0)
