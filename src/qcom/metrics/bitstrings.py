"""
Utilities for working with dictionaries keyed by bit strings.

Conventions
-----------
- Bit indexing uses **MSB ↔ index 0** (leftmost character in the bitstring).
  For example, for key "1010":
      index 0 → '1' (MSB)
      index 1 → '0'
      index 2 → '1'
      index 3 → '0' (LSB)

- Negative indices are accepted and normalized Python-style
  (e.g., -1 ≡ last bit, the LSB).
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Tuple

from qcom._internal.deprecations import warn_deprecated_alias

__all__ = [
    "sort_bitstring_distribution",
    "marginalize_bitstring_distribution",
    "order_dict",
    "part_dict",
]


def sort_bitstring_distribution(bitstring_values: Mapping[str, float]) -> Dict[str, float]:
    """
    Sort a bitstring-keyed distribution by integer bitstring value.

    Args:
        bitstring_values: Mapping where keys are non-empty binary strings.

    Returns:
        Ordered dictionary sorted by integer values of binary keys.

    Raises:
        TypeError: If `bitstring_values` is not a dictionary.
        ValueError: If any key is not a non-empty binary string.
    """
    if not isinstance(bitstring_values, dict):
        raise TypeError("sort_bitstring_distribution: 'bitstring_values' must be a dict.")

    for bitstring in bitstring_values.keys():
        if (
            not isinstance(bitstring, str)
            or len(bitstring) == 0
            or any(char not in "01" for char in bitstring)
        ):
            raise ValueError(
                "sort_bitstring_distribution: invalid binary key "
                f"{bitstring!r} (expected non-empty '0'/'1' string)."
            )

    ordered_items = sorted(bitstring_values.items(), key=lambda item: int(item[0], 2))
    return dict(ordered_items)


def marginalize_bitstring_distribution(
    bitstring_values: Mapping[str, float],
    indices: Iterable[int],
) -> Dict[str, float]:
    """
    Marginalize a bitstring-keyed distribution onto selected bit indices.

    For each key K in `bitstring_values`, this function constructs a new key K' by taking
    K[i] for i in `indices` (in the order provided), then sums values for any keys
    that reduce to the same K'.

    Args:
        bitstring_values: Mapping where keys are binary strings consisting of '0'/'1'.
        indices: Iterable of integer bit positions to extract **with 0 = MSB (leftmost)**.
                 Negative indices are allowed and are normalized Python-style
                 (e.g., -1 refers to the last bit, i.e., LSB).

    Returns:
        dict: New dictionary where keys contain only the extracted bits.
              Values are summed if multiple original keys reduce to the same substring.

    Raises:
        TypeError: If inputs have incorrect types.
        ValueError: If keys are invalid, indices are empty, out-of-bounds after normalization,
                    or contain duplicates.
    """
    if not isinstance(bitstring_values, dict):
        raise TypeError("marginalize_bitstring_distribution: 'bitstring_values' must be a dict.")
    try:
        indices_tuple: Tuple[int, ...] = tuple(int(i) for i in indices)
    except Exception:
        raise TypeError(
            "marginalize_bitstring_distribution: 'indices' must be an iterable of integers."
        ) from None
    if len(indices_tuple) == 0:
        raise ValueError("marginalize_bitstring_distribution: 'indices' cannot be empty.")

    lengths = set()
    for bitstring in bitstring_values.keys():
        if (
            not isinstance(bitstring, str)
            or len(bitstring) == 0
            or any(char not in "01" for char in bitstring)
        ):
            raise ValueError(
                "marginalize_bitstring_distribution: invalid binary key "
                f"{bitstring!r} (expected non-empty '0'/'1' string)."
            )
        lengths.add(len(bitstring))

    if len(lengths) == 0:
        return {}
    if len(lengths) != 1:
        raise ValueError(
            "marginalize_bitstring_distribution: all keys must have equal length; "
            f"got lengths {sorted(lengths)}."
        )
    bitstring_length = lengths.pop()

    normalized_indices: Tuple[int, ...] = tuple(
        index if index >= 0 else bitstring_length + index for index in indices_tuple
    )
    if any(index < 0 or index >= bitstring_length for index in normalized_indices):
        raise ValueError(
            "marginalize_bitstring_distribution: indices out of bounds after normalization "
            f"for key length {bitstring_length}: {indices_tuple} -> {normalized_indices}."
        )
    if len(set(normalized_indices)) != len(normalized_indices):
        raise ValueError("marginalize_bitstring_distribution: duplicate indices are not allowed.")

    marginalized_values: Dict[str, float] = {}
    for bitstring, value in bitstring_values.items():
        extracted_bits = "".join(bitstring[index] for index in normalized_indices)
        marginalized_values[extracted_bits] = marginalized_values.get(extracted_bits, 0) + value

    return marginalized_values


def order_dict(inp_dict: Dict[str, float]) -> Dict[str, float]:
    """Deprecated compatibility alias for `sort_bitstring_distribution`."""
    warn_deprecated_alias("order_dict", "sort_bitstring_distribution")
    return sort_bitstring_distribution(inp_dict)


def part_dict(inp_dict: Dict[str, float], indices: Iterable[int]) -> Dict[str, float]:
    """Deprecated compatibility alias for `marginalize_bitstring_distribution`."""
    warn_deprecated_alias("part_dict", "marginalize_bitstring_distribution")
    return marginalize_bitstring_distribution(inp_dict, indices)
