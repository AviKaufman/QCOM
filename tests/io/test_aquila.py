import json
from pathlib import Path

import pytest

from qcom.io.aquila import parse_aquila_json, parse_json


def _write_json(tmp_path: Path, payload: dict, name: str = "aquila.json") -> str:
    p = tmp_path / name
    p.write_text(json.dumps(payload), encoding="utf-8")
    return str(p)


def test_parse_aquila_json_inverts_bits_and_counts_when_filtering(tmp_path):
    """
    With sorted=True:
      - Only accept shots with preSequence all-ones.
      - Invert postSequence bits and form bitstrings.
    """
    payload = {
        "measurements": [
            # Accepted: preSequence is all ones; post=[0,1,1] -> invert -> [1,0,0] -> "100"
            {"shotResult": {"preSequence": [1, 1, 1], "postSequence": [0, 1, 1]}},
            {"shotResult": {"preSequence": [1, 0, 1], "postSequence": [1, 0, 1]}},
        ]
    }
    fn = _write_json(tmp_path, payload)

    counts, total = parse_aquila_json(fn, filter_incomplete=True, show_progress=False)

    assert total == pytest.approx(1.0)
    assert set(counts.keys()) == {"100"}
    assert isinstance(counts["100"], float)
    assert counts["100"] == pytest.approx(1.0)


def test_parse_aquila_json_includes_incomplete_when_filter_disabled(tmp_path):
    """
    With sorted=False:
      - Do not filter by preSequence completeness; both shots should count.
      - Check both bitstrings after inversion.
    """
    payload = {
        "measurements": [
            {"shotResult": {"preSequence": [1, 1, 1], "postSequence": [0, 1, 1]}},
            {"shotResult": {"preSequence": [1, 0, 1], "postSequence": [1, 0, 1]}},
        ]
    }
    fn = _write_json(tmp_path, payload)

    counts, total = parse_aquila_json(fn, filter_incomplete=False, show_progress=False)

    assert total == pytest.approx(2.0)
    assert counts == {"100": 1.0, "010": 1.0}


def test_parse_json_compatibility_alias_overrides_sorted(tmp_path):
    payload = {
        "measurements": [
            {"shotResult": {"preSequence": [1, 0], "postSequence": [0, 1]}},
        ]
    }
    fn = _write_json(tmp_path, payload)

    with pytest.warns(DeprecationWarning, match="parse_aquila_json"):
        counts, total = parse_json(fn, sorted=True, filter_incomplete=False, show_progress=False)

    assert total == pytest.approx(1.0)
    assert counts == {"10": 1.0}


def test_parse_aquila_json_handles_bad_records_gracefully(tmp_path, capsys):
    """
    If a record is malformed (e.g., missing 'shotResult'), the parser should
    print an error and continue processing the rest.
    """
    payload = {
        "measurements": [
            {"shotResult": {"preSequence": [1, 1], "postSequence": [1, 0]}},
            {"notShotResult": {"oops": True}},
            {"shotResult": {"preSequence": [1, 1], "postSequence": [0, 0]}},
        ]
    }
    fn = _write_json(tmp_path, payload)

    counts, total = parse_aquila_json(fn, filter_incomplete=False, show_progress=False)

    assert total == pytest.approx(2.0)
    assert counts == {"01": 1.0, "11": 1.0}

    captured = capsys.readouterr()
    assert "Error reading measurement" in captured.out
