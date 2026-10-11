"""Tests for code normalization and tokenization (SPEC 4.3)."""
from __future__ import annotations

from gcoderag.textnorm import extract_codes, normalize_code, normalize_codes, tokenize


def test_normalize_codes() -> None:
    assert normalize_codes("G01 X1") == "G1 X1"
    assert normalize_codes("M 03 S1000") == "M3 S1000"
    assert normalize_codes("G38.2 probe") == "G38.2 probe"
    assert normalize_codes("g00 then G90") == "G0 then G90"
    assert normalize_code("m30") == "M30"


def test_tokenize() -> None:
    assert tokenize("What does G1 do? Use G38.2.") == ["what", "does", "g1", "do", "use", "g38.2"]
    assert tokenize("") == []


def test_extract_codes_dedupes_in_order() -> None:
    assert extract_codes("G2, G3 and G02") == ["G2", "G3"]
