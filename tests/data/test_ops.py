# tests/test_ops.py
import math
import pytest

from qcom.data.ops import (
    normalize_to_probabilities,
    truncate_probabilities,
    print_most_probable_data,
)


# ------------------------------------------------------------------------------------
# normalize_to_probabilities
# ------------------------------------------------------------------------------------

def test_normalize_empty_ok_and_sum():
    counts = {}
    probs = normalize_to_probabilities(counts, total_count=10)
    assert probs == {}  # empty dict stays empty


def test_normalize_raises_on_missing_total():
    with pytest.raises(ValueError):
        _ = normalize_to_probabilities({"0": 1}, total_count=None)  # type: ignore[arg-type]


def test_normalize_raises_on_zero_total():
    with pytest.raises(ValueError):
        _ = normalize_to_probabilities({"0": 1}, total_count=0)

# ------------------------------------------------------------------------------------
# truncate_probabilities
# ------------------------------------------------------------------------------------

def test_truncate_keeps_equal_and_above_threshold():
    probs = {"00": 0.50, "01": 0.20, "10": 0.19, "11": 0.11}
    out = truncate_probabilities(probs, threshold=0.20)
    assert out == {"00": 0.50, "01": 0.20}


def test_truncate_all_removed_when_threshold_gt_all():
    probs = {"00": 0.4, "01": 0.6}
    out = truncate_probabilities(probs, threshold=0.99)
    assert out == {}


def test_truncate_empty_input_ok():
    assert truncate_probabilities({}, threshold=0.1) == {}


# ------------------------------------------------------------------------------------
# print_most_probable_data (stdout capture)
# ------------------------------------------------------------------------------------

def test_print_most_probable_top_n(capsys):
    probs = {"00": 0.1, "01": 0.4, "10": 0.3, "11": 0.2}
    print_most_probable_data(probs, n=3)

    captured = capsys.readouterr().out.strip().splitlines()
    # Header
    assert captured[0].strip() == "Top 3 Most probable bit strings:"

    # The function prints in descending probability order; check the first three.
    # Lines look like: "1.  Bit string: 01, Probability: 0.40000000"
    line1 = captured[1]
    line2 = captured[2]
    line3 = captured[3]

    assert "1." in line1 and "Bit string: 01" in line1 and "0.40000000" in line1
    assert "2." in line2 and "Bit string: 10" in line2 and "0.30000000" in line2
    assert "3." in line3 and "Bit string: 11" in line3 and "0.20000000" in line3


def test_print_most_probable_n_larger_than_dict(capsys):
    probs = {"0": 0.7, "1": 0.3}
    print_most_probable_data(probs, n=5)
    out = capsys.readouterr().out

    # Should still print header with requested n, and only two entries.
    assert "Top 5 Most probable bit strings:" in out
    assert out.count("Bit string: ") == 2