import os
import tempfile
import pandas as pd
import qcom as qc
import pytest
import json
import random

"""
This file is for testing the I/O functions in qcom/io.py using pytest. To run the tests, use the command 
`pytest tests/test_io` in the root directory of the repository. If you get an error, 
do not push the changes to GitHub until the error is fixed.
"""

# --- Tests for parse_file ---


def test_parse_file_basic():
    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmp:
        tmp.write("0000 1\n")
        tmp.write("1111 2\n")
        tmp.write("1010 3\n")
        file_path = tmp.name

    try:
        data, total = qc.parse_file(file_path)

        # Expected values
        expected_data = {"0000": 1.0, "1111": 2.0, "1010": 3.0}
        expected_total = 6.0

        assert data == expected_data
        assert total == expected_total

    finally:
        os.remove(file_path)  # Clean up the temp file


def test_parse_file_skips_blank_lines(tmp_path):
    # blank lines and whitespace-only lines should be skipped without error
    content = "\n0000 1\n\n   \n1111 2\n"
    fp = tmp_path / "data.txt"
    fp.write_text(content)

    data, total = qc.parse_file(str(fp), show_progress=False)

    assert total == pytest.approx(3.0)
    assert data == {"0000": 1.0, "1111": 2.0}


def test_parse_file_handles_malformed_line(tmp_path, capsys):
    # malformed line should print an error but not raise
    content = "0000 1\nbad_line_without_space\n1111 2\n"
    fp = tmp_path / "data.txt"
    fp.write_text(content)

    data, total = qc.parse_file(str(fp), show_progress=False)

    # Should have skipped the bad line but processed the two valid ones
    assert total == pytest.approx(3.0)
    assert data == {"0000": 1.0, "1111": 2.0}

    captured = capsys.readouterr()
    assert "Error reading line 'bad_line_without_space'" in captured.out


# --- Tests for parse_parq ---


def test_parse_parq_basic():
    # Create a simple DataFrame to write as a Parquet file
    df = pd.DataFrame(
        {"state": ["0000", "1111", "1010"], "probability": [0.1, 0.2, 0.7]}
    )

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        file_path = tmp.name
        df.to_parquet(file_path, engine="pyarrow")

    try:
        parsed = qc.parse_parq(file_path)

        expected = {"0000": 0.1, "1111": 0.2, "1010": 0.7}

        assert parsed == expected

    finally:
        os.remove(file_path)


def test_parse_parq_empty(tmp_path):
    # an empty DataFrame round-trips to an empty dict
    df = pd.DataFrame({"state": [], "probability": []})
    fp = tmp_path / "empty.parquet"
    df.to_parquet(fp, engine="pyarrow")

    parsed = qc.parse_parq(str(fp), show_progress=False)
    assert parsed == {}


def test_parse_parq_duplicate_states(tmp_path):
    # later rows override earlier when states are duplicated
    df = pd.DataFrame(
        {
            "state": ["0000", "0000", "1111"],
            "probability": [0.1, 0.5, 0.2],
        }
    )
    fp = tmp_path / "dup.parquet"
    df.to_parquet(fp, engine="pyarrow")

    parsed = qc.parse_parq(str(fp), show_progress=False)
    # the second "0000" (0.5) should have replaced the first
    assert parsed == {"0000": 0.5, "1111": 0.2}


def test_parse_parq_missing_file():
    # nonexistent file should raise FileNotFoundError
    with pytest.raises(FileNotFoundError):
        qc.parse_parq("no_such_file.parquet")


# --- Helpers for parse_json tests ---


def write_json(tmp_path, measurements):
    """
    Write a minimal JSON file with only the 'measurements' key.
    """
    p = tmp_path / "data.json"
    p.write_text(json.dumps({"measurements": measurements}))
    return str(p)


# --- parse_json tests, assuming show_progress=False ---


def test_parse_json_sorted(tmp_path):
    measurements = [
        {"shotResult": {"preSequence": [1, 1, 1], "postSequence": [0, 1, 0]}},
        {"shotResult": {"preSequence": [1, 0, 1], "postSequence": [1, 1, 1]}},
    ]
    fp = write_json(tmp_path, measurements)

    data, total = qc.parse_json(fp, sorted=True, show_progress=False)

    # Only the first line passes the all-1s preSequence check.
    # postSequence [0,1,0] → invert → [1,0,1] → "101"
    assert total == 1
    assert data == {"101": 1}


def test_parse_json_unsorted(tmp_path):
    measurements = [
        {"shotResult": {"preSequence": [1, 1, 1], "postSequence": [0, 1, 0]}},
        {"shotResult": {"preSequence": [1, 0, 1], "postSequence": [0, 0, 1]}},
    ]
    fp = write_json(tmp_path, measurements)

    data, total = qc.parse_json(fp, sorted=False, show_progress=False)

    # Both kept: [0,1,0]→"101" and [0,0,1]→"110"
    assert total == 2
    assert data == {"101": 1, "110": 1}


def test_parse_json_duplicate_counts(tmp_path):
    measurements = [
        {"shotResult": {"preSequence": [1, 1], "postSequence": [0, 0]}},
        {"shotResult": {"preSequence": [1, 1], "postSequence": [0, 0]}},
    ]
    fp = write_json(tmp_path, measurements)

    data, total = qc.parse_json(fp, sorted=True, show_progress=False)

    # Two identical shots → invert both to "11"
    assert total == 2
    assert data == {"11": 2}


# --- Tests for save_data ---


def test_save_data_basic():
    data = {"0000": 1.0, "1111": 2.0, "1010": 3.0}

    with tempfile.NamedTemporaryFile(mode="r+", delete=False) as tmp:
        filepath = tmp.name

    try:
        qc.save_data(data, filepath)

        with open(filepath, "r") as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]

        expected_lines = ["0000 1.0", "1111 2.0", "1010 3.0"]
        assert sorted(lines) == sorted(expected_lines)

    finally:
        os.remove(filepath)


def test_save_data_empty(tmp_path):
    """Saving an empty dict should produce an empty file."""
    filepath = tmp_path / "empty.txt"
    qc.save_data({}, str(filepath), show_progress=False)

    # File exists but is empty
    assert filepath.exists()
    assert filepath.read_text() == ""


def test_save_data_order(tmp_path):
    """Insertion order of the dict should be preserved in the output."""
    data = {"first": 1.1, "second": 2.2, "third": 3.3}
    filepath = tmp_path / "ordered.txt"
    qc.save_data(data, str(filepath), show_progress=False)

    lines = filepath.read_text().splitlines()
    assert lines == [
        "first 1.1",
        "second 2.2",
        "third 3.3",
    ]


# --- Tests for save_dict_to_parquet ---


def test_save_dict_to_parquet_basic():
    data_dict = {"0000": 0.1, "1111": 0.2, "1010": 0.7}

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        filepath = tmp.name

    try:
        qc.save_dict_to_parquet(data_dict, filepath)

        df = pd.read_parquet(filepath, engine="pyarrow")
        loaded_dict = dict(zip(df["state"], df["probability"]))

        assert loaded_dict == data_dict

    finally:
        os.remove(filepath)


def test_save_dict_to_parquet_empty(tmp_path):
    """An empty dict should produce an empty Parquet file (round-trip to {})."""
    fp = tmp_path / "empty.parquet"
    qc.save_dict_to_parquet({}, str(fp))

    # File should exist and be readable
    df = pd.read_parquet(str(fp), engine="pyarrow")
    # No rows, no columns beyond 'state' and 'probability'
    assert list(df.columns) == ["state", "probability"]
    assert df.shape[0] == 0


def test_save_dict_to_parquet_order(tmp_path):
    """Insertion order of the dict should be preserved in the 'state' column."""
    data_dict = {"a": 0.1, "bb": 0.2, "ccc": 0.3}
    fp = tmp_path / "ordered.parquet"
    qc.save_dict_to_parquet(data_dict, str(fp))

    df = pd.read_parquet(str(fp), engine="pyarrow")
    # The 'state' column should match insertion order
    assert list(df["state"]) == ["a", "bb", "ccc"]
    # And probabilities line up
    assert list(df["probability"]) == [0.1, 0.2, 0.3]


if __name__ == "__main__":
    # Run the tests from only this file
    pytest.main([__file__])
