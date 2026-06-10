# tests/test_sampling.py
import math
import random
import pytest

from qcom.core import CountsData, ProbabilityData
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


def test_combine_raw_dict_count_total_one_uses_legacy_probability_heuristic():
    # Raw dict compatibility still treats sum≈1 as probabilities.
    out = combine_datasets({"0": 1}, {"0": 1})

    assert out == {"0": 1.0}


def test_combine_counts_data_total_one_stays_counts():
    out = combine_datasets(CountsData({"0": 1}), CountsData({"0": 1}), return_data=True)

    assert isinstance(out, CountsData)
    assert out.to_dict() == {"0": 2}
    assert out.shots == 2


def test_combine_probability_data_is_explicit_and_normalized():
    out = combine_datasets(
        ProbabilityData({"0": 0.7, "1": 0.3}),
        ProbabilityData({"0": 0.2, "1": 0.8}),
        return_data=True,
    )

    assert isinstance(out, ProbabilityData)
    assert math.isclose(sum(out.probabilities.values()), 1.0)
    assert math.isclose(out.probabilities["0"], 0.45)
    assert math.isclose(out.probabilities["1"], 0.55)
