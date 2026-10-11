"""G/M code normalization and tokenization shared by indexing and queries (SPEC 4.3)."""
from __future__ import annotations

import re

CODE_RE = re.compile(r"\b([GgMm])\s?0*(\d+(?:\.\d+)?)\b")
_TOKEN_SPLIT_RE = re.compile(r"[^a-z0-9.]+")


def _format_code(letter: str, number: str) -> str:
    """Return the canonical spelling of a code: upper-case letter, no leading zeros."""
    integer, _, fraction = number.partition(".")
    integer = integer.lstrip("0") or "0"
    return f"{letter.upper()}{integer}" + (f".{fraction}" if fraction else "")


def normalize_code(code: str) -> str:
    """Normalize a single code such as 'g01' or 'M 03' to 'G1' / 'M3'; return it unchanged if it is not one."""
    match = CODE_RE.fullmatch(code.strip())
    return _format_code(match.group(1), match.group(2)) if match else code.strip()


def normalize_codes(text: str) -> str:
    """Rewrite every G/M code in `text` to canonical form: 'g01' -> 'G1', 'M 03' -> 'M3', 'G38.2' unchanged."""
    return CODE_RE.sub(lambda m: _format_code(m.group(1), m.group(2)), text)


def extract_codes(text: str) -> list[str]:
    """Return the normalized G/M codes found in `text`, deduplicated in order of appearance."""
    seen: dict[str, None] = {}
    for match in CODE_RE.finditer(text):
        seen.setdefault(_format_code(match.group(1), match.group(2)), None)
    return list(seen)


def tokenize(text: str) -> list[str]:
    """Lowercase and split on characters outside [a-z0-9.]; drop empties and strip trailing dots."""
    tokens = (tok.rstrip(".") for tok in _TOKEN_SPLIT_RE.split(text.lower()))
    return [tok for tok in tokens if tok and tok != "."]
