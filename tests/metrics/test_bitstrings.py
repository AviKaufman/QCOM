# tests/test_metrics_bitstrings.py
import pytest
from qcom.metrics.bitstrings import order_dict, part_dict


# -------------------- order_dict tests --------------------

def test_order_dict_sorts_by_integer_value():
    inp = {"10": 0.1, "01": 0.2, "00": 0.3, "11": 0.4}
    ordered = order_dict(inp)
    keys = list(ordered.keys())
    # "00"(0), "01"(1), "10"(2), "11"(3)
    assert keys == ["00", "01", "10", "11"]


def test_order_dict_invalid_key_type():
    with pytest.raises(ValueError):
        order_dict({"2A": 1.0})  # non-binary chars


def test_order_dict_empty_key_raises():
    with pytest.raises(ValueError):
        order_dict({"": 1.0})


def test_order_dict_type_error_non_dict():
    with pytest.raises(TypeError):
        order_dict([("01", 1.0)])


# -------------------- part_dict tests --------------------

def test_part_dict_extract_single_index():
    inp = {"00": 1.0, "01": 2.0, "10": 3.0, "11": 4.0}
    # Extract MSB (index 0)
    reduced = part_dict(inp, [0])
    # Should sum values by MSB
    assert reduced == {"0": 1.0 + 2.0, "1": 3.0 + 4.0}


def test_part_dict_extract_multiple_indices_and_order():
    inp = {"101": 1.0, "111": 2.0}
    # Extract indices [0,2] → first and last bits
    reduced = part_dict(inp, [0, 2])
    # "101" -> "11"; "111" -> "11"
    assert reduced == {"11": 3.0}


def test_part_dict_negative_indexing():
    inp = {"10": 5.0, "01": 7.0}
    # Index -1 is LSB
    reduced = part_dict(inp, [-1])
    assert reduced == {"0": 5.0, "1": 7.0}


def test_part_dict_duplicate_indices_error():
    inp = {"01": 1.0, "10": 2.0}
    with pytest.raises(ValueError, match="duplicate indices"):
        part_dict(inp, [0, 0])


def test_part_dict_out_of_bounds_index():
    inp = {"01": 1.0, "10": 2.0}
    with pytest.raises(ValueError, match="out of bounds"):
        part_dict(inp, [5])


def test_part_dict_mixed_length_keys_error():
    inp = {"0": 1.0, "01": 2.0}
    with pytest.raises(ValueError, match="equal length"):
        part_dict(inp, [0])


def test_part_dict_invalid_key_characters():
    inp = {"0X1": 1.0}
    with pytest.raises(ValueError):
        part_dict(inp, [0, 1, 2])


def test_part_dict_empty_indices_error():
    inp = {"00": 1.0}
    with pytest.raises(ValueError, match="cannot be empty"):
        part_dict(inp, [])


def test_part_dict_non_iterable_indices_type_error():
    inp = {"00": 1.0}
    with pytest.raises(TypeError):
        part_dict(inp, None)


def test_part_dict_non_dict_input_type_error():
    with pytest.raises(TypeError):
        part_dict([("01", 1.0)], [0])