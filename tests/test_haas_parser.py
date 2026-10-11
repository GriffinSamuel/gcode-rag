"""Tests for the Haas page parser on a small hand-written fixture (no network)."""
from __future__ import annotations

from pathlib import Path

from gcoderag.ingest.haas import parse_page

FIXTURE = Path(__file__).parent / "fixtures" / "haas_snippet.html"
URL = "https://example.invalid/mill---g-codes.html"


def test_parse_page_headings_and_table_rows() -> None:
    docs = parse_page(FIXTURE.read_text(encoding="utf-8"), "mill-gcodes", URL)
    ids = [d.doc_id for d in docs]
    assert ids[0] == "haas:mill-gcodes:g01-linear-interpolation-motion-group-01"
    assert docs[0].codes == ["G1"]
    assert "haas:mill-gcodes:g0-rapid-motion-positioning" in ids
    row = next(d for d in docs if d.title == "G12 Circular Pocket Milling CW")
    assert row.text == f"G12 Circular Pocket Milling CW (Group 00). See {URL}."
    assert all(d.source == "haas" for d in docs)
    assert not any(d.title == "Short" for d in docs)
