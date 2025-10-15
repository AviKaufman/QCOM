# tests/test_sampling.py
import math
import random
import pytest

from qcom.data.sampling import sample_data, combine_datasets


# ------------------------------------------------------------------------------------
# sample_data
# ------------------------------------------------------------------------------------

def test_sample_data_counts_sum_to_sample_size_and_is_reproducible():
    counts = {"00": 50, "01": 25, "10": 25}  # total_count = 100
    sample_size = 1000

    # Deterministic sampling via the global RNG used inside the module
    random.seed(12345)
    out1 = sample_data(counts, total_count=100, sample_size=sample_size, show_progress=False)

    random.seed(12345)
    out2 = sample_data(counts, total_count=100, sample_size=sample_size, show_progress=False)

    assert out1 == out2
    assert sum(out1.values()) == sample_size


def test_sample_data_raises_on_zero_total_count():
    counts = {"0": 2, "1": 1}
    with pytest.raises(ValueError):
        _ = sample_data(counts, total_count=0, sample_size=10)


def test_sample_data_raises_with_empty_population():
    # normalize_to_probabilities returns {}, then random.choices([], k>0) fails
    with pytest.raises(Exception):
        _ = sample_data({}, total_count=10, sample_size=5)


# ------------------------------------------------------------------------------------
# combine_datasets
# ------------------------------------------------------------------------------------

def test_combine_counts_adds_counts():
    d1 = {"00": 2, "01": 3}
    d2 = {"01": 1, "10": 4}
    out = combine_datasets(d1, d2, show_progress=False)

    assert out == {"00": 2, "01": 4, "10": 4}
    # counts remain counts (no renormalization)
    assert sum(out.values()) == sum(d1.values()) + sum(d2.values())


def test_combine_mixed_counts_and_probabilities_raises():
    probs = {"0": 0.7, "1": 0.3}
    counts = {"0": 7, "1": 3}
    with pytest.raises(ValueError):
        _ = combine_datasets(probs, counts)