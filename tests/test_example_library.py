"""The example library: every program is validator-clean, retrieval picks relevant ones, the generator uses them."""
from __future__ import annotations

import pytest

from gcoderag.embed import FakeEmbedder
from gcoderag.examples import EXAMPLES_PER_REQUEST_DEFAULT, ExampleLibrary, load_examples, parse_example
from gcoderag.generator import EXAMPLES_HEADER, GcodeGenerator
from gcoderag.llm import FakeLLM
from gcoderag.validator import GcodeValidator

VALID = """```gcode
G21 G90 G17
T1 M6
S10000 M3
G0 Z5
G1 Z-1 F200
G1 X20
G0 Z5
M5
M30
```"""


@pytest.fixture(scope="module")
def library() -> ExampleLibrary:
    return ExampleLibrary(load_examples(), FakeEmbedder())


def test_library_has_eight_clean_programs() -> None:
    examples = load_examples()
    assert len(examples) == 8
    assert sorted(e.dialect for e in examples) == ["haas"] * 2 + ["linuxcnc"] * 6
    for e in examples:
        report = GcodeValidator(e.dialect, use_corpus=False).validate(e.program)
        assert report.errors == [] and report.warnings == [], f"{e.name}: {report.summary()}"
        assert e.request and e.name.startswith(e.dialect)


def test_similar_returns_the_closest_examples(library) -> None:
    top = library.similar("peck drill three 6 mm holes 20 mm deep with 3 mm pecks", k=2)
    assert len(top) == 2 and top[0].name == "linuxcnc_peck_drill_g83"
    assert library.similar("engrave the letter L 0.2 mm deep with a V-bit", k=1)[0].name == "linuxcnc_engraving"
    assert library.similar("anything", k=3, dialect=None).__len__() == 3


def test_similar_prefers_the_requested_dialect(library) -> None:
    haas = library.similar("drill four holes in a square", k=2, dialect="haas")
    assert [e.dialect for e in haas] == ["haas", "haas"] and haas[0].name == "haas_drill_pattern"
    mixed = library.similar("face a block", k=3, dialect="haas")
    assert [e.dialect for e in mixed][:2] == ["haas", "haas"] and mixed[2].dialect == "linuxcnc"


def test_parse_example_requires_headers() -> None:
    ok = parse_example("x", "(dialect: haas)\n(request: Drill a hole)\nG0 X1\n")
    assert (ok.dialect, ok.request) == ("haas", "Drill a hole")
    with pytest.raises(ValueError):
        parse_example("bad", "G0 X1\n")


def test_empty_library_rejected() -> None:
    with pytest.raises(ValueError):
        ExampleLibrary([], FakeEmbedder())


def test_generator_puts_examples_before_the_request(toy_retriever, library) -> None:
    llm = FakeLLM([VALID])
    gen = GcodeGenerator(toy_retriever, llm, GcodeValidator(use_corpus=False), library=library)
    result = gen.generate("peck drill three 6 mm holes 20 mm deep with 3 mm pecks")
    user = llm.calls[0]["user"]
    assert result.report.ok and EXAMPLES_HEADER in user and user.count("```gcode") == EXAMPLES_PER_REQUEST_DEFAULT
    assert "G98 G83" in user
    assert user.index("Reference passages:") < user.index(EXAMPLES_HEADER) < user.index("Request: peck drill")


def test_generator_without_library_has_no_examples_block(toy_retriever) -> None:
    llm = FakeLLM([VALID])
    GcodeGenerator(toy_retriever, llm, GcodeValidator(use_corpus=False)).generate("cut a line")
    assert EXAMPLES_HEADER not in llm.calls[0]["user"]
