import pytest
from qcom.metrics.bitstrings import (
    marginalize_bitstring_distribution,
    order_dict,
    part_dict,
    sort_bitstring_distribution,
)


def test_sort_bitstring_distribution_sorts_by_integer_value():
    bitstring_values = {"10": 0.1, "01": 0.2, "00": 0.3, "11": 0.4}
    ordered = sort_bitstring_distribution(bitstring_values)
    keys = list(ordered.keys())
    # "00"(0), "01"(1), "10"(2), "11"(3)
    assert keys == ["00", "01", "10", "11"]


def test_sort_bitstring_distribution_invalid_key_type():
    with pytest.raises(ValueError):
        sort_bitstring_distribution({"2A": 1.0})


def test_sort_bitstring_distribution_empty_key_raises():
    with pytest.raises(ValueError):
        sort_bitstring_distribution({"": 1.0})


def test_sort_bitstring_distribution_type_error_non_dict():
    with pytest.raises(TypeError):
        sort_bitstring_distribution([("01", 1.0)])


def test_marginalize_bitstring_distribution_extract_single_index():
    bitstring_values = {"00": 1.0, "01": 2.0, "10": 3.0, "11": 4.0}
    reduced = marginalize_bitstring_distribution(bitstring_values, [0])
    assert reduced == {"0": 1.0 + 2.0, "1": 3.0 + 4.0}


def test_marginalize_bitstring_distribution_extract_multiple_indices_and_order():
    bitstring_values = {"101": 1.0, "111": 2.0}
    reduced = marginalize_bitstring_distribution(bitstring_values, [0, 2])
    # "101" -> "11"; "111" -> "11"
    assert reduced == {"11": 3.0}


def test_marginalize_bitstring_distribution_negative_indexing():
    bitstring_values = {"10": 5.0, "01": 7.0}
    reduced = marginalize_bitstring_distribution(bitstring_values, [-1])
    assert reduced == {"0": 5.0, "1": 7.0}


def test_marginalize_bitstring_distribution_duplicate_indices_error():
    bitstring_values = {"01": 1.0, "10": 2.0}
    with pytest.raises(ValueError, match="duplicate indices"):
        marginalize_bitstring_distribution(bitstring_values, [0, 0])


def test_marginalize_bitstring_distribution_out_of_bounds_index():
    bitstring_values = {"01": 1.0, "10": 2.0}
    with pytest.raises(ValueError, match="out of bounds"):
        marginalize_bitstring_distribution(bitstring_values, [5])


def test_marginalize_bitstring_distribution_mixed_length_keys_error():
    bitstring_values = {"0": 1.0, "01": 2.0}
    with pytest.raises(ValueError, match="equal length"):
        marginalize_bitstring_distribution(bitstring_values, [0])


def test_marginalize_bitstring_distribution_invalid_key_characters():
    bitstring_values = {"0X1": 1.0}
    with pytest.raises(ValueError):
        marginalize_bitstring_distribution(bitstring_values, [0, 1, 2])


def test_marginalize_bitstring_distribution_empty_indices_error():
    bitstring_values = {"00": 1.0}
    with pytest.raises(ValueError, match="cannot be empty"):
        marginalize_bitstring_distribution(bitstring_values, [])


def test_marginalize_bitstring_distribution_non_iterable_indices_type_error():
    bitstring_values = {"00": 1.0}
    with pytest.raises(TypeError):
        marginalize_bitstring_distribution(bitstring_values, None)


def test_marginalize_bitstring_distribution_non_dict_input_type_error():
    with pytest.raises(TypeError):
        marginalize_bitstring_distribution([("01", 1.0)], [0])


def test_bitstring_distribution_compatibility_aliases():
    bitstring_values = {"10": 1.0, "01": 2.0}
    with pytest.warns(DeprecationWarning, match="sort_bitstring_distribution"):
        assert order_dict(bitstring_values) == sort_bitstring_distribution(bitstring_values)
    with pytest.warns(DeprecationWarning, match="marginalize_bitstring_distribution"):
        assert part_dict(bitstring_values, [0]) == marginalize_bitstring_distribution(
            bitstring_values,
            [0],
        )
