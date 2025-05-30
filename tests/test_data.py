import random
import pytest
import numpy as np
import qcom as qc

"""
Tests for qcom/data.py functions.
Assumes show_progress=False (ProgressManager calls are no-ops).
To run: pytest tests/test_data.py
"""

# --- Tests for normalize_to_probabilities ---


def test_normalize_to_probabilities_none_total():
    with pytest.raises(ValueError):
        qc.normalize_to_probabilities({"a": 1}, None)


def test_normalize_to_probabilities_zero_total():
    with pytest.raises(ValueError):
        qc.normalize_to_probabilities({"a": 0, "b": 0}, 0)


def test_normalize_to_probabilities_success():
    data = {"x": 2, "y": 3}
    result = qc.normalize_to_probabilities(data, 5)
    assert np.isclose(result["x"], 2 / 5, atol=1e-12)
    assert np.isclose(result["y"], 3 / 5, atol=1e-12)


# --- Tests for sample_data ---


def test_sample_data_deterministic(monkeypatch):
    # Given fixed random.choices output, ensure sample_data counts match
    data = {"00": 10, "11": 30}
    total_count = 40
    sample_size = 5
    # stub random.choices to return a fixed sequence
    monkeypatch.setattr(
        random, "choices", lambda sequences, weights, k: ["00", "11", "11", "00", "11"]
    )
    result = qc.sample_data(data, total_count, sample_size, show_progress=False)
    # Should count occurrences exactly
    assert result == {"00": 2, "11": 3}
    # Total samples should equal sample_size
    assert sum(result.values()) == sample_size


def test_sample_data_keys_valid():
    # Without stubbing, ensure keys in result come from original data
    random.seed(0)
    data = {"a": 1, "b": 2, "c": 3}
    total_count = 6
    sample_size = 10
    result = qc.sample_data(data, total_count, sample_size, show_progress=False)
    for key in result:
        assert key in data
    # sum of result == sample_size
    assert sum(result.values()) == sample_size


# --- Tests for introduce_error ---


def test_introduce_error_no_flip(monkeypatch):
    # random.random always > rates → no flip
    monkeypatch.setattr(random, "random", lambda: 1.0)
    data = {"01": 5, "10": 5}
    result = qc.introduce_error(data, ground_rate=0.5, excited_rate=0.5)
    # Should equal original counts
    assert result == data


def test_introduce_error_all_flip(monkeypatch):
    # random.random always 0 → always flip
    monkeypatch.setattr(random, "random", lambda: 0.0)
    # '0' flips to '1', '1' flips to '0'
    data = {"0": 1, "1": 1}
    result = qc.introduce_error(data, ground_rate=1.0, excited_rate=1.0)
    # Expect one '1' (from '0') and one '0' (from '1')
    assert result == {"1": 1, "0": 1}


def test_introduce_error_counts_preserved(monkeypatch):
    # Ensure total counts preserved
    data = {"00": 2, "11": 3}
    total = sum(data.values())
    random.seed(1)
    result = qc.introduce_error(data, ground_rate=0.3, excited_rate=0.4)
    assert sum(result.values()) == total


# --- Tests for print_most_probable_data ---


def test_print_most_probable_data_basic(capsys):
    normalized_data = {"x": 0.6, "y": 0.3, "z": 0.1}
    qc.print_most_probable_data(normalized_data, n=2)
    out = capsys.readouterr().out.strip().splitlines()
    # Header
    assert out[0] == "Most probable 2 bit strings:"
    # There should be exactly n+1 lines
    assert len(out) == 3
    # Check ordering and formatting
    assert out[1].startswith("1.")
    assert "x" in out[1]
    assert out[2].startswith("2.")
    assert "y" in out[2]


def test_print_most_probable_data_ties(capsys):
    # Equal probabilities, ensure deterministic ordering by key order after sort
    normalized_data = {"a": 0.5, "b": 0.5}
    qc.print_most_probable_data(normalized_data, n=2)
    out = capsys.readouterr().out
    # Both should appear
    assert "a" in out and "b" in out


# --- Tests for combine_datasets ---


def test_combine_datasets_counts():
    data1 = {"a": 1, "b": 2}
    data2 = {"b": 3, "c": 4}
    result = qc.combine_datasets(data1, data2, show_progress=False)
    assert result == {"a": 1, "b": 5, "c": 4}


def test_combine_datasets_probabilities():
    data1 = {"a": 0.2, "b": 0.8}
    data2 = {"b": 0.1, "c": 0.9}
    combined = {"a": 0.2, "b": 0.9, "c": 0.9}
    total = sum(combined.values())
    expected = {k: v / total for k, v in combined.items()}
    result = qc.combine_datasets(data1, data2, show_progress=False)
    for key in expected:
        assert np.isclose(result[key], expected[key], atol=1e-8)


def test_combine_datasets_mixed():
    prob = {"a": 1.0}
    counts = {"a": 2}
    with pytest.raises(ValueError):
        qc.combine_datasets(prob, counts, show_progress=False)


def test_combine_datasets_empty():
    # Combining two empty dicts should yield empty
    assert qc.combine_datasets({}, {}, show_progress=False) == {}


# --- Tests for truncate_probabilities ---


def test_truncate_probabilities_basic():
    data = {"a": 0.05, "b": 0.5, "c": 0.95}
    res = qc.truncate_probabilities(data, 0.5)
    assert res == {"b": 0.5, "c": 0.95}


def test_truncate_probabilities_all():
    data = {"x": 0.2}
    assert qc.truncate_probabilities(data, 0.0) == data


def test_truncate_probabilities_none():
    data = {"x": 0.1}
    assert qc.truncate_probabilities(data, 1.0) == {}


if __name__ == "__main__":
    pytest.main([__file__])
