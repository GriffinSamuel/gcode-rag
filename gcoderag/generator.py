"""Generate G-code with an LLM, validate it, and retry on errors and warnings (SPEC 6.3)."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Literal

from gcoderag.examples import EXAMPLES_PER_REQUEST_DEFAULT, ExampleLibrary
from gcoderag.index import HybridRetriever
from gcoderag.llm import LLM
from gcoderag.rag import best_chunk_per_doc, format_passages
from gcoderag.validator import DEFAULT_ENVELOPE, GcodeValidator, Issue, ValidationReport

logger = logging.getLogger(__name__)

GEN_PASSAGES = 6
GEN_RETRIEVE_K = 20
GEN_MAX_TOKENS = 1200
FIX_PROMPT = "Fix these validation errors and return the complete corrected program:"
WARNING_PROMPT = ("The program is valid but has these warnings; fix them if possible and return the complete "
                  "program:")
EXAMPLES_HEADER = "Reference programs that passed validation:"
_FENCE_RE = re.compile(r"```[ \t]*([A-Za-z0-9_-]*)[^\n]*\n(.*?)```", re.DOTALL)

_COMMON_RULES = """Program rules (all are required):
- First line is a units line: G21 (mm) or G20 (inch), together with G90 (absolute) and G17 (XY plane).
- Tool change with T1 M6 before cutting.
- Start the spindle with S<rpm> M3 before any cutting move.
- Set a feed rate F before (or on) the first G1, G2 or G3 move.
- Retract to a safe Z (G0 Z5 or higher) before any XY rapid move; never rapid (G0) below Z0.
- Arcs (G2/G3) always give I/J or R.
- Use only one code from each modal group per line.
- Stay inside the machine envelope: X {x}, Y {y}, Z {z} mm.
- End with M5 and then M30.
- Output ONLY the program inside a ```gcode fenced block, with short ( ... ) comments. No prose."""
_DIALECT_RULES = {
    "linuxcnc": "Dialect: LinuxCNC (RS274/NGC). Do not use % or O program numbers.",
    "haas": "Dialect: Haas mill (Fanuc style). The first and last lines are % alone; the second line is the program "
            "number O00001. Haas-specific codes are allowed.",
}


@dataclass
class GenResult:
    """The best program found, its validation report, and every attempt made.

    `history` has one dict per LLM call: attempt, program, ok, errors/warnings (text), n_errors, n_warnings.
    """

    program: str
    report: ValidationReport
    attempts: int
    dialect: str
    history: list[dict[str, Any]]


def extract_program(text: str) -> str:
    """Return the first fenced code block (preferring ```gcode / ```nc), or the whole reply if there is none."""
    blocks = _FENCE_RE.findall(text)
    if not blocks:
        return text.strip()
    preferred = [body for lang, body in blocks if lang.lower() in {"gcode", "nc", "ngc", "g-code"}]
    return (preferred or [blocks[0][1]])[0].strip()


def system_prompt(dialect: str) -> str:
    """The generation system prompt with the dialect rules and envelope."""
    env = {axis.lower(): f"[{lo:g}, {hi:g}]" for axis, (lo, hi) in DEFAULT_ENVELOPE.items()}
    return ("You are a careful CNC programmer writing G-code for a 3-axis mill.\n"
            f"{_DIALECT_RULES[dialect]}\n{_COMMON_RULES.format(**env)}\n"
            "Use the reference passages to choose codes and their words correctly.")


def _issue_lines(issues: list[Issue]) -> str:
    return "\n".join(f"{i.rule} line {i.line_no}: {i.message} | {i.raw}" for i in issues)


def _quality(report: ValidationReport) -> tuple[int, int]:
    """Sort key for 'which result is better': fewer errors first, then fewer warnings."""
    return len(report.errors), len(report.warnings)


class GcodeGenerator:
    """LLM-written programs, checked by the deterministic validator, retried with errors and then warnings."""

    def __init__(self, retriever: HybridRetriever, llm: LLM | None, validator: GcodeValidator,
                 library: ExampleLibrary | None = None) -> None:
        self.retriever = retriever
        self.llm = llm
        self.validator = validator
        self.library = library

    def _validator_for(self, dialect: str) -> GcodeValidator:
        if dialect == self.validator.dialect:
            return self.validator
        v = self.validator
        return GcodeValidator(dialect, envelope=v.envelope, extra_codes=v.extra_codes, corpus_docs=v.corpus_docs,
                              use_corpus=v.use_corpus)

    def _examples_block(self, request: str, dialect: str) -> str:
        """'Reference programs that passed validation' for the most similar library examples ('' if no library)."""
        if self.library is None:
            return ""
        examples = self.library.similar(request, k=EXAMPLES_PER_REQUEST_DEFAULT, dialect=dialect)
        if not examples:
            return ""
        shown = "\n\n".join(f"Request: {e.request}\n```gcode\n{e.program}\n```" for e in examples)
        return f"{EXAMPLES_HEADER}\n\n{shown}\n\n"

    def generate(self, request: str, dialect: Literal["linuxcnc", "haas"] = "linuxcnc",
                 max_attempts: int = 3) -> GenResult:
        """Generate a program for `request`.

        Each LLM call is one attempt. A program with errors is retried with the error list. A program with no errors
        but with warnings gets ONE extra request asking to fix the warnings (if attempts remain). The result returned
        is the best attempt by (errors, warnings), so a clean program is never lost to a worse retry.
        """
        if self.llm is None:
            raise RuntimeError("LLM not configured")
        if dialect not in _DIALECT_RULES:
            raise ValueError(f"unknown dialect {dialect!r}")
        validator = self._validator_for(dialect)
        passages = best_chunk_per_doc(self.retriever.retrieve(request, k=GEN_RETRIEVE_K, mode="hybrid"), GEN_PASSAGES)
        base_user = (f"Reference passages:\n\n{format_passages(passages)}\n\n"
                     f"{self._examples_block(request, dialect)}Request: {request}")
        system = system_prompt(dialect)
        user = base_user
        history: list[dict[str, Any]] = []
        best: tuple[str, ValidationReport] | None = None
        warning_request_sent = False
        attempts = 0
        for attempt in range(1, max_attempts + 1):
            attempts = attempt
            reply = self.llm.complete(system, user, max_tokens=GEN_MAX_TOKENS, temperature=0.0)
            program = extract_program(reply)
            report = validator.validate(program)
            history.append({"attempt": attempt, "program": program, "ok": report.ok,
                            "errors": [f"{i.rule} line {i.line_no}: {i.message}" for i in report.errors],
                            "warnings": [f"{i.rule} line {i.line_no}: {i.message}" for i in report.warnings],
                            "n_errors": len(report.errors), "n_warnings": len(report.warnings)})
            if best is None or _quality(report) < _quality(best[1]):
                best = (program, report)
            if not report.ok:
                logger.info("attempt %d failed validation with %d error(s)", attempt, len(report.errors))
                feedback = f"{FIX_PROMPT}\n{_issue_lines(report.errors)}"
            elif report.warnings and not warning_request_sent and attempt < max_attempts:
                warning_request_sent = True
                feedback = f"{WARNING_PROMPT}\n{_issue_lines(report.warnings)}"
            else:
                break
            user = f"{base_user}\n\nYour previous program:\n```gcode\n{program}\n```\n\n{feedback}"
        assert best is not None
        return GenResult(best[0], best[1], attempts, dialect, history)
