"""Tests for the LinuxCNC AsciiDoc parser (SPEC 3.3) on the fixture snippet."""
from __future__ import annotations

from pathlib import Path

import pytest

from gcoderag.ingest.linuxcnc import clean_title, heading_codes, parse_adoc

FIXTURE = Path(__file__).parent / "fixtures" / "linuxcnc_snippet.adoc"


@pytest.fixture(scope="module")
def docs():
    return parse_adoc(FIXTURE.read_text(encoding="utf-8"), "g-code")


def test_three_docs_with_expected_ids(docs) -> None:
    assert [d.doc_id for d in docs] == ["linuxcnc:sec-conventions", "linuxcnc:gcode:g4", "linuxcnc:gcode:g10-l2"]


def test_titles_codes_and_urls(docs) -> None:
    g4, g10 = docs[1], docs[2]
    assert g4.title == "G4 Dwell"
    assert g4.codes == ["G4"]
    assert g4.url == "https://linuxcnc.org/docs/stable/html/gcode/g-code.html#gcode:g4"
    assert g4.section_path == "g-code > G4 Dwell"
    assert g10.codes == ["G10"]
    assert docs[0].codes == []


def test_text_cleaning(docs) -> None:
    g4, g10 = docs[1], docs[2]
    assert g4.text.startswith("G4 Dwell\n\n")
    assert "G4 P0.5 (wait for 0.5 seconds" in g4.text  # source blocks are kept
    assert "comment block" not in g4.text
    assert "[source" not in g4.text and "----" not in g4.text
    assert "See also G1 for feed moves." in g4.text  # cross-reference converted
    assert "G10 L2 P1 | Set the G54 origin" in g10.text  # table cells joined
    assert "image::" not in g10.text and "|===" not in g10.text
    assert "line comment" not in docs[0].text


def test_short_sections_are_skipped(docs) -> None:
    assert all("Tiny" != d.title for d in docs)


def test_title_and_heading_codes() -> None:
    assert clean_title("G2, G3 Arc Move(((G2, G3 Arc Move)))") == "G2, G3 Arc Move"
    assert heading_codes("M3, M4, M5 Spindle Control") == ["M3", "M4", "M5"]
    assert heading_codes("G01 Linear") == ["G1"]
