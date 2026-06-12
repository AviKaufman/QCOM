import math
import random
import pytest

from qcom.core import CountsData, ProbabilityData
from qcom.data.sampling import (
    combine_bitstring_datasets,
    combine_datasets,
    sample_counts,
    sample_data,
)


def test_sample_counts_sum_to_sample_size_and_is_reproducible():
    counts = {"00": 50, "01": 25, "10": 25}  # total_count = 100
    sample_size = 1000

    random.seed(12345)
    out1 = sample_counts(counts, total_count=100, sample_size=sample_size, show_progress=False)

    random.seed(12345)
    out2 = sample_counts(counts, total_count=100, sample_size=sample_size, show_progress=False)

    assert out1 == out2
    assert sum(out1.values()) == sample_size


def test_sample_counts_raises_on_zero_total_count():
    counts = {"0": 2, "1": 1}
    with pytest.raises(ValueError):
        _ = sample_counts(counts, total_count=0, sample_size=10)


def test_sample_counts_raises_with_empty_population():
    with pytest.raises(Exception):
        _ = sample_counts({}, total_count=10, sample_size=5)


def test_combine_counts_adds_counts():
    d1 = {"00": 2, "01": 3}
    d2 = {"01": 1, "10": 4}
    out = combine_bitstring_datasets(d1, d2, show_progress=False)

    assert out == {"00": 2, "01": 4, "10": 4}
    assert sum(out.values()) == sum(d1.values()) + sum(d2.values())


def test_combine_mixed_counts_and_probabilities_raises():
    probs = {"0": 0.7, "1": 0.3}
    counts = {"0": 7, "1": 3}
    with pytest.raises(ValueError):
        _ = combine_bitstring_datasets(probs, counts)


def test_combine_raw_dict_count_total_one_uses_legacy_probability_heuristic():
    out = combine_bitstring_datasets({"0": 1}, {"0": 1})

    assert out == {"0": 1.0}


def test_combine_counts_data_total_one_stays_counts():
    out = combine_bitstring_datasets(CountsData({"0": 1}), CountsData({"0": 1}), return_data=True)

    assert isinstance(out, CountsData)
    assert out.to_dict() == {"0": 2}
    assert out.shots == 2


def test_combine_probability_data_is_explicit_and_normalized():
    out = combine_bitstring_datasets(
        ProbabilityData({"0": 0.7, "1": 0.3}),
        ProbabilityData({"0": 0.2, "1": 0.8}),
        return_data=True,
    )

    assert isinstance(out, ProbabilityData)
    assert math.isclose(sum(out.probabilities.values()), 1.0)
    assert math.isclose(out.probabilities["0"], 0.45)
    assert math.isclose(out.probabilities["1"], 0.55)


def test_sampling_compatibility_aliases():
    random.seed(12345)
    expected = sample_counts({"0": 1}, total_count=1, sample_size=3)
    random.seed(12345)
    with pytest.warns(DeprecationWarning, match="sample_counts"):
        assert sample_data({"0": 1}, total_count=1, sample_size=3) == expected
    with pytest.warns(DeprecationWarning, match="combine_bitstring_datasets"):
        alias_combined = combine_datasets({"0": 2}, {"0": 3})
    assert alias_combined == combine_bitstring_datasets(
        {"0": 2},
        {"0": 3},
    )
