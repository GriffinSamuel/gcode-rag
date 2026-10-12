"""Tests for the rule-based G-code validator: one per rule plus parsing and state tracking."""
from __future__ import annotations

import json

import pytest

from gcoderag.validator import GcodeValidator, ValidationReport, detect_dialect, main, parse_program

HEADER = "G21 G90 G17\nT1 M6\nS1000 M3\nF300\n"
FOOTER = "M5\nM30\n"
VALID = """(dialect: linuxcnc)
G21 G90 G17      ; metric, absolute, XY plane
T1 M6
S12000 M3
G0 Z5
G0 X0 Y0
G1 Z-1 F150
G1 X20 Y0 F400
G2 X20 Y20 I0 J10
G1 X0 Y20
G0 Z5
M5
M30
"""
HAAS_VALID = """%
O00001 (dialect: haas)
G20 G90 G17
T1 M06
S3000 M03
G00 Z0.2
G01 Z-0.05 F20.
G01 X2.
G00 Z1.
M05
M30
%
"""


def validate(text: str, dialect: str = "linuxcnc", **kwargs) -> ValidationReport:
    return GcodeValidator(dialect, use_corpus=False, **kwargs).validate(text)


def rules(report: ValidationReport) -> set[str]:
    return {i.rule for i in report.errors + report.warnings}


def program(body: str) -> str:
    return HEADER + body + "\n" + FOOTER


def test_valid_program_has_no_errors_or_warnings() -> None:
    report = validate(VALID)
    assert report.ok and report.errors == [] and report.warnings == []
    assert report.blocks == 12


def test_e001_stray_characters() -> None:
    report = validate(program("G1 X10 $ Y5"))
    assert not report.ok and "E001" in rules(report)


def test_e002_letter_without_number_and_bad_value() -> None:
    assert "E002" in rules(validate(program("G1 X")))
    assert "E002" in rules(validate(program("G1 X1.2.3")))


def test_e003_unknown_code_depends_on_dialect() -> None:
    assert "E003" in rules(validate(program("G1 X5\nM13")))
    assert "E003" not in rules(validate(program("G1 X5\nM13"), dialect="haas"))
    assert "E003" in rules(validate(program("G999 X1")))


def test_e003_extra_codes_and_corpus_codes(tmp_path) -> None:
    assert "E003" not in rules(validate(program("G1 X5\nM13"), extra_codes={"M13"}))
    docs = tmp_path / "docs.jsonl"
    docs.write_text(json.dumps({"source": "linuxcnc", "codes": ["G43.4"]}) + "\n", encoding="utf-8")
    report = GcodeValidator(corpus_docs=docs).validate(program("G43.4 H1"))
    assert "E003" not in rules(report)


def test_e004_modal_group_conflicts() -> None:
    assert "E004" in rules(validate(program("G0 G1 X10")))
    assert "E004" in rules(validate(program("G90 G91 X10")))
    assert "E004" in rules(validate(program("M8 M9")))
    assert "E004" not in rules(validate(program("M7 M8\nM9")))


def test_e005_feed_move_without_feed() -> None:
    report = validate("G21 G90\nS1000 M3\nG1 X10\nG1 X20\nM30\n")
    e005 = [i for i in report.errors if i.rule == "E005"]
    assert len(e005) == 1 and e005[0].line_no == 3
    assert "E005" not in rules(validate("G21 G90\nS1000 M3\nG1 X10 F100\nM30\n"))


def test_e005_modal_feed_motion_without_g_word() -> None:
    report = validate("G21 G90\nS1000 M3\nG1 F0.0001\nX10\nM30\n")
    assert "E005" not in rules(report)
    report = validate("G21 G90\nS1000 M3\nG1\nX10\nM30\n")
    assert "E005" in rules(report)


def test_e006_arc_without_center_or_radius() -> None:
    assert "E006" in rules(validate(program("G2 X10 Y10")))
    assert "E006" not in rules(validate(program("G2 X10 Y10 R5")))
    assert "E006" not in rules(validate(program("G3 X10 Y10 I5 J0")))


def test_e007_tool_change_without_tool() -> None:
    assert "E007" in rules(validate("G21 G90\nM6\nM30\n"))
    assert "E007" not in rules(validate("G21 G90\nT2\nM6\nM30\n"))


def test_e008_spindle_start_without_speed() -> None:
    assert "E008" in rules(validate("G21 G90\nM3\nM30\n"))
    assert "E008" not in rules(validate("G21 G90\nM3 S500\nM30\n"))


def test_e009_envelope_absolute() -> None:
    report = validate(program("G0 X500"))
    assert any(i.rule == "E009" and "X moves to 500" in i.message for i in report.errors)
    assert "E009" in rules(validate(program("G1 Z-300")))
    assert "E009" not in rules(validate(program("G1 X400 Y-400 Z-250")))


def test_e009_envelope_tracking_in_g91() -> None:
    body = "G91\n" + "\n".join("G1 X100" for _ in range(4)) + "\nG90"
    assert "E009" not in rules(validate(program(body)))
    body = "G91\n" + "\n".join("G1 X100" for _ in range(5)) + "\nG90"
    report = validate(program(body))
    assert [i.line_no for i in report.errors if i.rule == "E009"] == [10]


def test_e009_inch_units_are_converted() -> None:
    assert "E009" in rules(validate("G20 G90\nS500 M3\nG1 X16 F10\nM30\n"))      # 406.4 mm
    assert "E009" not in rules(validate("G20 G90\nS500 M3\nG1 X15 F10\nM30\n"))  # 381 mm


def test_e009_ignores_offsets_and_homing_axis_words() -> None:
    assert "E009" not in rules(validate(program("G10 L2 P1 X900\nG28 G91 Z999\nG90")))


def test_e010_non_positive_feed() -> None:
    assert "E010" in rules(validate(program("G1 X10 F0")))
    assert "E010" in rules(validate(program("G1 X10 F-5")))


def test_w101_missing_program_end() -> None:
    report = validate("G21 G90\nS1000 M3\nG1 X10 F100\n")
    assert report.ok and "W101" in rules(report)


def test_w102_feed_move_with_spindle_off() -> None:
    report = validate("G21 G90\nG1 X10 F100\nM30\n")
    assert report.ok and "W102" in rules(report)
    assert "W102" in rules(validate("G21 G90\nS100 M3\nM5\nG1 X10 F100\nM30\n"))


def test_w103_rapid_plunge_below_zero() -> None:
    assert "W103" in rules(validate(program("G0 Z5\nG0 Z-2")))
    assert "W103" not in rules(validate(program("G0 Z5\nG0 Z1\nG1 Z-2")))


def test_w104_haas_percent_and_program_number() -> None:
    assert validate(HAAS_VALID, "haas").warnings == []
    no_percent = HAAS_VALID.replace("%\n", "")
    assert "W104" in rules(validate(no_percent, "haas"))
    no_o = HAAS_VALID.replace("O00001 ", "")
    assert "W104" in rules(validate(no_o, "haas"))
    assert "W104" not in rules(validate(no_percent, "linuxcnc"))


def test_w105_missing_units() -> None:
    report = validate("G90\nS1000 M3\nG1 X10 F100\nM30\n")
    assert report.ok and "W105" in rules(report)


def test_w106_parameters_are_skipped() -> None:
    report = validate(program("#100 = 5\nG1 X#100"))
    assert report.ok and [i.rule for i in report.warnings] == ["W106", "W106"]


def test_haas_percent_lines_are_skipped_blocks() -> None:
    blocks = parse_program(HAAS_VALID)
    assert blocks[0].skipped and blocks[0].raw == "%" and blocks[-1].skipped
    assert validate(HAAS_VALID, "haas").ok


def test_ocode_and_block_delete_lines_are_skipped() -> None:
    text = program("o100 sub\nG1 X1\no100 endsub\no100 call\n/G1 X999 $$\no<probe> call")
    report = validate(text)
    assert report.ok and report.warnings == []
    assert [b.skipped for b in parse_program("o100 sub\n/G1 X1\nO1234\n")] == [True, True, False]


def test_comments_are_stripped() -> None:
    blocks = parse_program("G1 X10 (move $ right) Y5 ; trailing $ comment\n(only a comment)\n")
    assert len(blocks) == 1
    assert [(w.letter, w.value) for w in blocks[0].words] == [("G", 1.0), ("X", 10.0), ("Y", 5.0)]
    assert blocks[0].parse_issues == []


def test_code_normalization_g01_equals_g1() -> None:
    report = validate(program("G01 X1\nG00 G01 X2"))
    assert any(i.rule == "E004" and "G0 and G1" in i.message for i in report.errors)


def test_summary_and_to_dict() -> None:
    report = validate(program("G1 X10 F0"))
    lines = report.summary().splitlines()
    assert lines[0].startswith("INVALID (linuxcnc): 1 error(s)")
    assert lines[1] == "E010 line 5: feed rate F0 must be greater than 0 | G1 X10 F0"
    assert json.loads(json.dumps(report.to_dict()))["errors"][0]["rule"] == "E010"


def test_detect_dialect() -> None:
    assert detect_dialect("(dialect: haas)\nG0 X0") == "haas"
    assert detect_dialect("(Dialect: LinuxCNC)") == "linuxcnc"
    assert detect_dialect("G0 X0") is None


def test_unknown_dialect_rejected() -> None:
    with pytest.raises(ValueError):
        GcodeValidator("fanuc")


def test_cli_exit_codes(tmp_path, capsys) -> None:
    good = tmp_path / "good.nc"
    good.write_text(VALID, encoding="utf-8")
    bad = tmp_path / "bad.nc"
    bad.write_text(program("G1 X10 F0"), encoding="utf-8")
    assert main([str(good)]) == 0
    assert main([str(bad), "--json"]) == 1
    assert '"rule": "E010"' in capsys.readouterr().out


def test_haas_canned_cycles_have_their_own_modal_group() -> None:
    safe_start = HAAS_VALID.replace("G20 G90 G17", "G00 G17 G40 G49 G80 G90\nG20")
    assert validate(safe_start, "haas").ok
    assert "E004" in rules(validate(program("G0 G80 X1")))                 # LinuxCNC: same group 1
    assert "E004" in rules(validate(program("G81 G83 X1 Z-1 R1"), "haas"))  # still one cycle per block
    report = validate("%\nO1\nG20 G90\nS500 M3\nG80 G01 X1.\nM30\n%\n", "haas")
    assert "E005" in rules(report)                                        # G1 (not G80) sets the motion mode
