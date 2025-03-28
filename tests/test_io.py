import os
import tempfile
import pandas as pd
import qcom as qc
import pytest

"""
This file is for testing the I/O functions in qcom/io.py using pytest. To run the tests, use the command 
`pytest tests/test_io` in the root directory of the repository. If you get an error, 
do not push the changes to GitHub until the error is fixed.
"""


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


if __name__ == "__main__":
    # Run the tests from only this file
    pytest.main([__file__])
