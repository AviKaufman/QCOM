import random
import pytest
import numpy as np
import qcom as qc

"""
This file is for testing the data manipulation functions in qcom/data.py using pytest. To run the tests,
use the command `pytest tests/test_data.py` in the root directory of the repository.
If you get an error, do not push the changes to GitHub until the error is fixed.
"""

# --- Tests for normalize_to_probabilities --- #


def test_normalize_to_probabilities_success():
    data = {"a": 2, "b": 3}
    total_count = 5
    expected = {"a": 2 / 5, "b": 3 / 5}
    result = qc.normalize_to_probabilities(data, total_count)
    for key in expected:
        assert np.isclose(result[key], expected[key], atol=1e-10)


def test_normalize_to_probabilities_zero_total():
    data = {"a": 0, "b": 0}
    total_count = 0
    with pytest.raises(ValueError):
        qc.normalize_to_probabilities(data, total_count)


# --- Tests for sample_data --- #


def test_sample_data_no_progress():
    random.seed(42)  # Set seed for reproducibility
    data = {"101": 100, "010": 200}
    total_count = 300
    sample_size = 1000
    result = qc.sample_data(data, total_count, sample_size, show_progress=False)
    # Verify that all keys in the result are from the original data.
    for key in result.keys():
        assert key in data
    # Check that the returned probabilities sum to 1.
    assert np.isclose(sum(result.values()), 1.0, atol=1e-6)
    # Optionally, check that frequencies are close to expected values.
    # Expected probabilities: "101": ~0.3333, "010": ~0.6667.
    assert np.isclose(result["101"], 100 / 300, atol=0.05)
    assert np.isclose(result["010"], 200 / 300, atol=0.05)


def test_sample_data_with_progress(capsys):
    random.seed(42)
    data = {"101": 100, "010": 200}
    total_count = 300
    sample_size = 100
    result = qc.sample_data(
        data, total_count, sample_size, update_interval=20, show_progress=True
    )
    captured = capsys.readouterr().out
    # Check that progress messages (e.g., "Sampling data") are printed.
    assert "Sampling data" in captured
    # Verify normalization.
    assert np.isclose(sum(result.values()), 1.0, atol=1e-6)


# --- Tests for introduce_error_data --- #


def test_introduce_error_data_no_error():
    # Set error rates to 0 so that no bits flip.
    data = {"101": 10, "010": 20}
    total_count = 30
    result = qc.introduce_error_data(
        data, total_count, ground_rate=0, excited_rate=0, show_progress=False
    )
    # Since normalized_data is computed from unique keys, each key is processed once.
    # Thus, regardless of original counts, we expect two keys with equal weight.
    assert len(result) == 2
    for prob in result.values():
        assert np.isclose(prob, 0.5, atol=1e-6)


def test_introduce_error_data_all_error():
    # With error rates 1, every bit will flip.
    # For a sequence like "101", flipping each bit yields "010".
    data = {"101": 10}
    total_count = 10
    result = qc.introduce_error_data(
        data, total_count, ground_rate=1, excited_rate=1, show_progress=False
    )
    # Expect that "101" becomes "010" deterministically.
    assert "010" in result
    assert np.isclose(result["010"], 1.0, atol=1e-6)


def test_introduce_error_data_with_progress(capsys):
    random.seed(42)
    data = {"101": 10, "010": 20}
    total_count = 30
    result = qc.introduce_error_data(
        data, total_count, update_interval=1, show_progress=True
    )
    captured = capsys.readouterr().out
    assert "Introducing errors" in captured
    assert np.isclose(sum(result.values()), 1.0, atol=1e-6)


# --- Test for print_most_probable_data --- #


def test_print_most_probable_data(capsys):
    normalized_data = {"101": 0.5, "010": 0.3, "110": 0.2}
    qc.print_most_probable_data(normalized_data, n=2)
    captured = capsys.readouterr().out
    # Check header and presence of the top two bit strings.
    assert "Most probable 2 bit strings:" in captured
    assert "101" in captured
    assert "010" in captured


# --- Tests for combine_datasets --- #


def test_combine_datasets_counts():
    # Both datasets are counts (sums not equal to 1), so no normalization is performed.
    data1 = {"101": 2, "010": 3}
    data2 = {"101": 1, "110": 4}
    result = qc.combine_datasets(data1, data2, show_progress=False)
    expected = {"101": 3, "010": 3, "110": 4}
    assert result == expected


def test_combine_datasets_probabilities():
    # Both datasets are probabilities (sums ≈ 1), so the result is renormalized.
    data1 = {"101": 0.4, "010": 0.6}
    data2 = {"101": 0.3, "110": 0.7}
    # Combined (unnormalized): {"101": 0.7, "010": 0.6, "110": 0.7}
    total = 0.7 + 0.6 + 0.7
    expected = {"101": 0.7 / total, "010": 0.6 / total, "110": 0.7 / total}
    result = qc.combine_datasets(data1, data2, show_progress=False)
    for key in expected:
        assert np.isclose(result[key], expected[key], atol=1e-6)


def test_combine_datasets_mixed():
    # One dataset is probabilities and the other is counts. This should raise a ValueError.
    data1 = {"101": 0.4, "010": 0.6}  # Probabilities (sum = 1)
    data2 = {"101": 2, "110": 4}  # Counts (sum != 1)
    with pytest.raises(ValueError):
        qc.combine_datasets(data1, data2, show_progress=False)


def test_combine_datasets_with_progress(capsys):
    data1 = {"101": 2, "010": 3}
    data2 = {"101": 1, "110": 4}
    result = qc.combine_datasets(data1, data2, update_interval=1, show_progress=True)
    captured = capsys.readouterr().out
    assert "Combining datasets" in captured
    expected = {"101": 3, "010": 3, "110": 4}
    assert result == expected


if __name__ == "__main__":
    pytest.main([__file__])
