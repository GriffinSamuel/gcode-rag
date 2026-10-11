"""Tests for the chunker (SPEC 3.4)."""
from __future__ import annotations

from gcoderag.ingest import Doc
from gcoderag.ingest.chunker import chunk_doc


def _doc(text: str) -> Doc:
    return Doc(doc_id="linuxcnc:gcode:test", source="linuxcnc", title="T1 Test", codes=["G1"],
               url="https://example.invalid#t", section_path="g-code > T1 Test", text=text)


def test_short_doc_is_one_chunk() -> None:
    chunks = chunk_doc(_doc("T1 Test\n\nShort body with a few words."))
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "linuxcnc:gcode:test#0"
    assert chunks[0].text.startswith("T1 Test")
    assert chunks[0].codes == ["G1"]


def test_paragraphs_pack_up_to_target() -> None:
    para = " ".join(["word"] * 100)
    chunks = chunk_doc(_doc("\n\n".join([para] * 5)), target_words=250)
    assert [len(c.text.split()) for c in chunks] == [200, 200, 100]
    assert [c.chunk_index for c in chunks] == [0, 1, 2]
    assert chunks[2].chunk_id.endswith("#2")


def test_long_paragraph_splits_with_overlap() -> None:
    words = [f"w{i}" for i in range(1000)]
    chunks = chunk_doc(_doc("T1 Test\n\n" + " ".join(words)), target_words=250, max_words=400, overlap_words=40)
    assert len(chunks) >= 4
    assert all(len(c.text.split()) <= 400 for c in chunks)
    first, second = chunks[1].text.split(), chunks[2].text.split()
    assert first[-40:] == second[:40]  # neighbouring pieces share 40 words
    assert chunks[0].text.startswith("T1 Test")


def test_long_paragraph_prefers_sentence_boundaries() -> None:
    sentence = " ".join(["alpha"] * 49) + " end."
    chunks = chunk_doc(_doc(" ".join([sentence] * 10)), target_words=250, max_words=400, overlap_words=0)
    assert all(c.text.endswith("end.") for c in chunks)
