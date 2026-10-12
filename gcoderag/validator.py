"""Deterministic rule-based G-code validator (SPEC 7). No LLM, no network.

Run as `python -m gcoderag.validator <file.nc> [--dialect linuxcnc|haas] [--json]`; exits 0 when the program has
no errors, else 1.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

DIALECTS = ("linuxcnc", "haas")
DEFAULT_ENVELOPE: dict[str, tuple[float, float]] = {"X": (-400.0, 400.0), "Y": (-400.0, 400.0), "Z": (-250.0, 250.0)}
MM_PER_INCH = 25.4


def _codes(letter: str, spec: str) -> frozenset[str]:
    """Expand a space-separated code list; `a-b` expands an integer range (e.g. "110-129")."""
    out: set[str] = set()
    for item in spec.split():
        if "-" in item:
            lo, hi = item.split("-")
            out.update(f"{letter}{n}" for n in range(int(lo), int(hi) + 1))
        else:
            out.add(f"{letter}{item}")
    return frozenset(out)


KNOWN_G_LINUXCNC = _codes("G", (
    "0 1 2 3 4 5 5.1 5.2 5.3 7 8 10 17 17.1 18 18.1 19 19.1 20 21 28 28.1 30 30.1 33 33.1 38.2 38.3 38.4 38.5 40 41 "
    "41.1 42 42.1 43 43.1 43.2 49 52 53 54 55 56 57 58 59 59.1 59.2 59.3 61 61.1 64 70 71 71.1 71.2 72 72.1 72.2 73 74 "
    "76 80 81 82 83 84 85 86 87 88 89 90 90.1 91 91.1 92 92.1 92.2 92.3 93 94 95 96 97 98 99"))
KNOWN_M_LINUXCNC = _codes("M", (
    "0 1 2 3 4 5 6 7 8 9 19 30 48 49 50 51 52 53 60 61 62 63 64 65 66 67 68 70 71 72 73 98 99 100-199"))
KNOWN_G_HAAS_EXTRA = _codes("G", (
    "9 12 13 47 50 51 60 65 100 101 103 107 110-129 136 141 143 150 153 154 174 184 187 188 234 253 254 255 266 268 "
    "269"))
KNOWN_M_HAAS_EXTRA = _codes("M", (
    "10 11 12 13 16 17 18 21 22 23 24 25 26 27 28 29 31 33 34 35 36 39 41 42 46 50 51-58 59 69 75 76 77 78 79 80 81 "
    "82 83 84 85 86 88 89 95 96 97 98 99 109 130 131 138 139 158 159"))

MODAL_GROUPS: dict[str, frozenset[str]] = {
    "G motion": _codes("G", "0 1 2 3 5 5.1 5.2 33 33.1 38.2 38.3 38.4 38.5 73 76 80 81 82 83 84 85 86 87 88 89 12 13 "
                            "150"),
    "G plane": _codes("G", "17 18 19 17.1 18.1 19.1"),
    "G distance": _codes("G", "90 91"),
    "G arc distance": _codes("G", "90.1 91.1"),
    "G feed mode": _codes("G", "93 94 95"),
    "G units": _codes("G", "20 21"),
    "G cutter comp": _codes("G", "40 41 42 41.1 42.1"),
    "G tool length": _codes("G", "43 43.1 43.2 49"),
    "G coord sys": _codes("G", "54 55 56 57 58 59 59.1 59.2 59.3 110-129 154"),
    "G return mode": _codes("G", "98 99"),
    "G path control": _codes("G", "61 61.1 64"),
    "G spindle mode": _codes("G", "96 97"),
    "G lathe mode": _codes("G", "7 8"),
    "M stopping": _codes("M", "0 1 2 30 60"),
    "M spindle": _codes("M", "3 4 5"),
    "M override": _codes("M", "48 49"),
}
FEED_MOTION = _codes("G", "1 2 3 33 73 76 81-89 12 13 150")
CANNED_CYCLES = _codes("G", "73 76 81-89")
POSITION_MOTION = _codes("G", "0 1 2 3") | CANNED_CYCLES
ARC_MOTION = _codes("G", "2 3")
# Non-modal codes whose axis words are not a programmed tool position (offsets, homing, machine coordinates).
AXIS_CONSUMERS = _codes("G", "10 28 28.1 30 30.1 52 53 92 92.1 92.2 92.3")
AXES = ("X", "Y", "Z")

_COMMENT_RE = re.compile(r"\([^)]*\)")
_OCODE_RE = re.compile(
    r"^[oO]\d*\s*(sub|endsub|call|if|elseif|else|endif|while|endwhile|do|break|continue|return|repeat|endrepeat)\b"
    r"|^[oO]<")
_TOKEN_RE = re.compile(r"\s*(?:([A-Za-z])\s*([-+]?[0-9.]*)|(\S))")
_DIALECT_RE = re.compile(r"\(\s*dialect\s*:\s*(\w+)\s*\)", re.IGNORECASE)


@dataclass
class Word:
    """One address word, e.g. `G1` or `X-10.5`."""

    letter: str
    value: float
    raw: str


@dataclass
class Block:
    """One program line after comment stripping. `skipped` blocks are not checked by the modal rules."""

    line_no: int
    raw: str
    words: list[Word]
    skipped: bool
    parse_issues: list[tuple[str, str]] = field(default_factory=list)   # (rule id, message)


@dataclass
class Issue:
    """A rule violation (error) or a style/safety concern (warning)."""

    rule: str
    line_no: int
    message: str
    raw: str
    severity: Literal["error", "warning"]


@dataclass
class ValidationReport:
    """Validation outcome: `ok` is True when there are no errors (warnings allowed)."""

    errors: list[Issue]
    warnings: list[Issue]
    ok: bool
    dialect: str
    blocks: int

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable form."""
        return asdict(self)

    def summary(self) -> str:
        """Human-readable report, one issue per line: `E005 line 12: ... | G1 X10`."""
        status = "VALID" if self.ok else "INVALID"
        head = (f"{status} ({self.dialect}): {len(self.errors)} error(s), {len(self.warnings)} warning(s), "
                f"{self.blocks} block(s)")
        lines = [f"{i.rule} line {i.line_no}: {i.message} | {i.raw}" for i in self.errors + self.warnings]
        return "\n".join([head, *lines])


def code_name(letter: str, value: float) -> str:
    """Normalized code text: ('G', 1.0) -> 'G1', ('G', 38.2) -> 'G38.2'."""
    return f"{letter}{value:g}"


def _parse_words(line: str) -> tuple[list[Word], list[tuple[str, str]]]:
    """Split a comment-free line into words, reporting E001 (stray characters) and E002 (missing/bad number)."""
    words: list[Word] = []
    issues: list[tuple[str, str]] = []
    for m in _TOKEN_RE.finditer(line):
        letter, number, stray = m.group(1), m.group(2), m.group(3)
        if stray is not None:
            issues.append(("E001", f"stray character {stray!r}"))
            continue
        if letter is None:
            continue
        raw = m.group(0).strip()
        if not number or number in {"+", "-"}:
            issues.append(("E002", f"letter {letter.upper()} has no number"))
            continue
        try:
            words.append(Word(letter.upper(), float(number), raw))
        except ValueError:
            issues.append(("E002", f"value {number!r} for {letter.upper()} is not a number"))
    return words, issues


def strip_comments(line: str) -> str:
    """Remove `( ... )` comments and everything after `;`, then trim."""
    return _COMMENT_RE.sub(" ", line).split(";", 1)[0].strip()


def parse_program(text: str) -> list[Block]:
    """Parse a program into blocks; blank and comment-only lines are dropped. Pure function."""
    blocks: list[Block] = []
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = strip_comments(raw_line)
        raw = raw_line.strip()
        if not line:
            continue
        if line == "%" or line.startswith("/") or _OCODE_RE.match(line):
            blocks.append(Block(line_no, raw, [], True))
        elif "#" in line or "[" in line:
            blocks.append(Block(line_no, raw, [], True, [("W106", "parameters or expressions are not validated")]))
        else:
            words, issues = _parse_words(line)
            blocks.append(Block(line_no, raw, words, False, issues))
    return blocks


def detect_dialect(text: str) -> str | None:
    """Return the dialect named in a `(dialect: haas)` comment, if any."""
    m = _DIALECT_RE.search(text)
    return m.group(1).lower() if m and m.group(1).lower() in DIALECTS else None


def load_corpus_codes(docs_path: Path, dialect: str) -> set[str]:
    """Codes named in corpus documents usable by `dialect` (LinuxCNC docs always; Haas docs for haas)."""
    if not docs_path.exists():
        return set()
    sources = {"linuxcnc", "haas"} if dialect == "haas" else {"linuxcnc"}
    codes: set[str] = set()
    with docs_path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                doc = json.loads(line)
                if doc.get("source") in sources:
                    codes.update(doc.get("codes") or [])
    return codes


@dataclass
class _State:
    """Modal state tracked while walking one program (local to a single `validate` call)."""

    motion: str | None = None
    incremental: bool = False
    inches: bool = False
    feed_set: bool = False
    tool_seen: bool = False
    speed_seen: bool = False
    spindle_on: bool = False
    pos: dict[str, float] = field(default_factory=lambda: {a: 0.0 for a in AXES})
    reported: set[str] = field(default_factory=set)


class GcodeValidator:
    """Checks a program against rules E001-E010 (errors) and W101-W106 (warnings)."""

    def __init__(self, dialect: str = "linuxcnc", envelope: dict[str, tuple[float, float]] | None = None,
                 extra_codes: set[str] | None = None, corpus_docs: Path | None = None,
                 use_corpus: bool = True) -> None:
        if dialect not in DIALECTS:
            raise ValueError(f"unknown dialect {dialect!r}; expected one of {DIALECTS}")
        self.dialect = dialect
        self.envelope = dict(envelope or DEFAULT_ENVELOPE)
        self.extra_codes = set(extra_codes or ())
        self._corpus_docs = corpus_docs
        self._use_corpus = use_corpus
        self._known: frozenset[str] | None = None

    @property
    def known_codes(self) -> frozenset[str]:
        """Built-in codes for the dialect, plus corpus codes (loaded lazily) and `extra_codes`."""
        if self._known is None:
            known = set(KNOWN_G_LINUXCNC | KNOWN_M_LINUXCNC) | self.extra_codes
            if self.dialect == "haas":
                known |= KNOWN_G_HAAS_EXTRA | KNOWN_M_HAAS_EXTRA
            if self._use_corpus:
                known |= load_corpus_codes(self._docs_path(), self.dialect)
            self._known = frozenset(known)
        return self._known

    def _docs_path(self) -> Path:
        if self._corpus_docs is not None:
            return self._corpus_docs
        from gcoderag.config import get_paths

        return get_paths().corpus / "docs.jsonl"

    def validate(self, text: str) -> ValidationReport:
        """Validate a whole program and return the report."""
        blocks = parse_program(text)
        issues: list[Issue] = []
        state = _State()
        for block in blocks:
            issues.extend(Issue(rule, block.line_no, msg, block.raw, _severity(rule))
                          for rule, msg in block.parse_issues)
            if not block.skipped:
                issues.extend(self._check_block(block, state))
        issues.extend(self._program_warnings(text, blocks))
        errors = [i for i in issues if i.severity == "error"]
        warnings = [i for i in issues if i.severity == "warning"]
        return ValidationReport(errors, warnings, not errors, self.dialect, len(blocks))

    def _check_block(self, block: Block, state: _State) -> list[Issue]:
        """Apply the per-block rules in LinuxCNC execution order (F, S, T before M6/M3, motion last)."""
        found: list[tuple[str, str]] = []
        codes = [code_name(w.letter, w.value) for w in block.words if w.letter in "GM"]
        letters = {w.letter for w in block.words}
        found += [("E003", f"unknown code {c} for dialect {self.dialect}") for c in codes if c not in self.known_codes]
        found += _modal_conflicts(codes)
        found += _check_words(block, state)
        found += _check_tool_and_spindle(codes, letters, state)
        found += self._check_motion(block, codes, letters, state)
        return [Issue(rule, block.line_no, msg, block.raw, _severity(rule)) for rule, msg in found]

    def _check_motion(self, block: Block, codes: list[str], letters: set[str], state: _State) -> list[tuple[str, str]]:
        """Modal updates, then E005/E006/E009/W102/W103 for the block's motion."""
        found: list[tuple[str, str]] = []
        state.incremental = _modal_flag(codes, "G91", "G90", state.incremental)
        state.inches = _modal_flag(codes, "G20", "G21", state.inches)
        motions = [c for c in codes if c in MODAL_GROUPS["G motion"]]
        if motions:
            state.motion = motions[-1]
        has_axes = bool(letters & set("XYZABCUVW"))
        if any(c in AXIS_CONSUMERS for c in codes):
            return found
        moving = bool(motions) or has_axes
        if moving and state.motion in FEED_MOTION:
            if not state.feed_set:
                found += _once(state, "E005", f"{state.motion} feed move but no F feed rate has been set")
            if not state.spindle_on:
                found += _once(state, "W102", f"{state.motion} feed move while the spindle is off")
        if moving and state.motion in ARC_MOTION and not letters & set("IJKR"):
            found.append(("E006", f"{state.motion} arc without I, J, K or R"))
        if has_axes and state.motion in POSITION_MOTION:
            found += self._move(block, state)
        return found

    def _move(self, block: Block, state: _State) -> list[tuple[str, str]]:
        """Update the tracked position (mm); report E009 outside the envelope and W103 rapid plunges."""
        found: list[tuple[str, str]] = []
        scale = MM_PER_INCH if state.inches else 1.0
        for w in block.words:
            if w.letter not in AXES:
                continue
            value = w.value * scale
            new = state.pos[w.letter] + value if state.incremental else value
            lo, hi = self.envelope.get(w.letter, (float("-inf"), float("inf")))
            if not lo <= new <= hi:
                found.append(("E009", f"{w.letter} moves to {new:g} mm, outside the envelope [{lo:g}, {hi:g}]"))
            if w.letter == "Z" and state.motion == "G0" and new < state.pos["Z"] and new < 0:
                found.append(("W103", f"rapid G0 plunge to Z{new:g} below 0"))
            state.pos[w.letter] = new
        return found

    def _program_warnings(self, text: str, blocks: list[Block]) -> list[Issue]:
        """Whole-program warnings W101, W104, W105."""
        codes = {code_name(w.letter, w.value) for b in blocks for w in b.words if w.letter in "GM"}
        out: list[Issue] = []
        if not codes & {"M2", "M30"}:
            out.append(Issue("W101", 0, "program has no M2 or M30 end", "", "warning"))
        if not codes & {"G20", "G21"}:
            out.append(Issue("W105", 0, "program never sets units with G20 or G21", "", "warning"))
        if self.dialect == "haas":
            lines = [strip_comments(ln) for ln in text.splitlines()]
            lines = [ln for ln in lines if ln]
            wrapped = bool(lines) and lines[0] == "%" and lines[-1] == "%"
            has_o = any(w.letter == "O" for b in blocks for w in b.words)
            if not wrapped or not has_o:
                out.append(Issue("W104", 0, "Haas program should start and end with % and have an O<number>",
                                 "", "warning"))
        return out


def _severity(rule: str) -> Literal["error", "warning"]:
    return "error" if rule.startswith("E") else "warning"


def _once(state: _State, rule: str, message: str) -> list[tuple[str, str]]:
    """Report `rule` only the first time it fires in a program (avoids one issue per line)."""
    if rule in state.reported:
        return []
    state.reported.add(rule)
    return [(rule, message + " (reported once per program)")]


def _modal_flag(codes: list[str], on: str, off: str, current: bool) -> bool:
    if on in codes:
        return True
    if off in codes:
        return False
    return current


def _modal_conflicts(codes: list[str]) -> list[tuple[str, str]]:
    """E004: two codes from one modal group in a block (M9 conflicts with M7/M8; M7+M8 is allowed)."""
    found = []
    for group, members in MODAL_GROUPS.items():
        hits = list(dict.fromkeys(c for c in codes if c in members))
        if len(hits) > 1:
            found.append(("E004", f"{' and '.join(hits)} are both in the {group} modal group"))
    if "M9" in codes and {"M7", "M8"} & set(codes):
        found.append(("E004", "M9 (coolant off) in the same block as M7/M8 (coolant on)"))
    return found


def _check_words(block: Block, state: _State) -> list[tuple[str, str]]:
    """E010 (F <= 0) and the F/S/T bookkeeping that later rules depend on."""
    found = []
    for w in block.words:
        if w.letter == "F":
            if w.value <= 0:
                found.append(("E010", f"feed rate F{w.value:g} must be greater than 0"))
            else:
                state.feed_set = True
        elif w.letter == "S":
            state.speed_seen = True
        elif w.letter == "T":
            state.tool_seen = True
    return found


def _check_tool_and_spindle(codes: list[str], letters: set[str], state: _State) -> list[tuple[str, str]]:
    """E007 (M6 without T), E008 (M3/M4 without S), and spindle on/off tracking."""
    found = []
    if "M6" in codes and not state.tool_seen:
        found.append(("E007", "M6 tool change but no T tool number has been given"))
    if {"M3", "M4"} & set(codes):
        if not state.speed_seen:
            found.append(("E008", "spindle start without an S speed"))
        state.spindle_on = True
    if {"M5", "M2", "M30"} & set(codes):
        state.spindle_on = False
    return found


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: validate one file, print the summary (or JSON), exit 0 if ok else 1."""
    parser = argparse.ArgumentParser(prog="python -m gcoderag.validator", description="Validate a G-code file")
    parser.add_argument("file")
    parser.add_argument("--dialect", choices=DIALECTS, default="linuxcnc")
    parser.add_argument("--json", action="store_true", help="print the full report as JSON")
    args = parser.parse_args(argv)
    try:
        text = Path(args.file).read_text(encoding="utf-8")
    except OSError as exc:
        print(f"cannot read {args.file}: {exc}", file=sys.stderr)
        return 2
    report = GcodeValidator(args.dialect).validate(text)
    print(json.dumps(report.to_dict(), indent=2) if args.json else report.summary())
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
