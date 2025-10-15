# tests/test_io_text.py
from pathlib import Path
import io
import os
import pytest

from qcom.io.text import parse_file, save_data


def test_roundtrip_save_and_parse(tmp_path: Path):
    data = {"00": 1.5, "01": 0.5, "10": 2.0}
    fn = tmp_path / "counts.txt"

    # Write
    save_data(data, str(fn), show_progress=False)

    # Read back
    parsed, total = parse_file(str(fn), show_progress=False)

    # Dict equality (order-insensitive)
    assert parsed == data
    # Totals as sum of values
    assert total == pytest.approx(sum(data.values()))
    # Types: ensure floats
    assert all(isinstance(v, float) for v in parsed.values())


def test_parse_aggregates_duplicate_states(tmp_path: Path):
    # Manually craft a file with repeated keys
    content = "\n".join(
        [
            "000 1.0",
            "111 2.0",
            "000 0.5",  # duplicate key should accumulate
            "010 3.25",
        ]
    )
    fn = tmp_path / "dupes.txt"
    fn.write_text(content)

    parsed, total = parse_file(str(fn), show_progress=False)
    assert parsed["000"] == pytest.approx(1.5)
    assert parsed["111"] == pytest.approx(2.0)
    assert parsed["010"] == pytest.approx(3.25)
    assert total == pytest.approx(1.5 + 2.0 + 3.25)


def test_parse_ignores_blank_and_reports_malformed_lines(tmp_path: Path, capsys: pytest.CaptureFixture):
    # Include blank lines and a malformed line (missing value)
    content = "\n".join(
        [
            "",                 # blank
            "000 1.0",
            "bad_line",         # malformed (no split into 2 parts)
            "111 not_a_float",  # malformed (value error)
            "010 2.5",
            "   ",              # whitespace-only
        ]
    )
    fn = tmp_path / "messy.txt"
    fn.write_text(content)

    parsed, total = parse_file(str(fn), show_progress=False)

    # Only valid lines counted
    assert parsed == {"000": 1.0, "010": 2.5}
    assert total == pytest.approx(3.5)

    # Ensure warnings were printed for malformed lines
    out = capsys.readouterr().out
    assert "Error reading line" in out
    # We expect both malformed lines to be mentioned
    assert "bad_line" in out
    assert "not_a_float" in out


def test_parse_nonexistent_file_raises():
    with pytest.raises(FileNotFoundError):
        parse_file("this/file/does/not/exist.txt", show_progress=False)


def test_save_overwrites_existing_file(tmp_path: Path):
    fn = tmp_path / "overwrite.txt"
    # Create an existing file
    fn.write_text("garbage\n")

    data = {"11": 4.0}
    save_data(data, str(fn), show_progress=False)

    # Parse should return only the new data
    parsed, total = parse_file(str(fn), show_progress=False)
    assert parsed == {"11": 4.0}
    assert total == pytest.approx(4.0)