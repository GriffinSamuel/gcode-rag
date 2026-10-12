"""Every examples/*.nc file validates to the outcome its filename prefix promises."""
from __future__ import annotations

from pathlib import Path

import pytest

from gcoderag.validator import GcodeValidator, detect_dialect

EXAMPLES = sorted((Path(__file__).resolve().parent.parent / "examples").glob("*.nc"))


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_example_matches_filename(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    dialect = detect_dialect(text)
    assert dialect is not None, "first comment must name the dialect"
    report = GcodeValidator(dialect, use_corpus=False).validate(text)
    assert report.ok == path.name.startswith("valid_"), report.summary()
