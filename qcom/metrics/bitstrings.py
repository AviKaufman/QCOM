# qcom/metrics/bitstrings.py
"""
qcom.metrics.bitstrings
=======================

Utilities for working with dictionaries keyed by bit strings.

Purpose
-------
- Provide basic operations for organizing, re-indexing, and manipulating
  dictionaries where keys are binary strings (e.g., measurement outcomes).
- Functions are lightweight and avoid external dependencies.

Functions
---------
- order_dict(inp_dict): order dictionary entries by integer interpretation of bit strings.
- part_dict(inp_dict, indices): extract/reduce bit strings to selected indices, aggregating values.

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
from typing import Dict, Iterable, Tuple


# -------------------- Order dictionary by integer value --------------------

def order_dict(inp_dict: Dict[str, float]) -> Dict[str, float]:
    """
    Orders a dictionary based on binary keys interpreted as integers.

    Args:
        inp_dict: Dictionary where keys are binary strings consisting of '0'/'1'.

    Returns:
        dict: Ordered dictionary sorted by integer values of binary keys.

    Raises:
        ValueError: If any key is not a non-empty binary string.
    """
    if not isinstance(inp_dict, dict):
        raise TypeError("order_dict: 'inp_dict' must be a dict.")

    for k in inp_dict.keys():
        if not isinstance(k, str) or len(k) == 0 or any(c not in "01" for c in k):
            raise ValueError(f"order_dict: invalid binary key {k!r} (expected non-empty '0'/'1' string).")

    ordered_items = sorted(inp_dict.items(), key=lambda item: int(item[0], 2))
    return dict(ordered_items)


# -------------------- Extract subset of bits --------------------

def part_dict(inp_dict: Dict[str, float], indices: Iterable[int]) -> Dict[str, float]:
    """
    Extracts a subset of bits from each binary string based on given indices (MSB=0 convention).

    For each key K in `inp_dict`, this function constructs a new key K' by taking
    K[i] for i in `indices` (in the order provided), then sums values for any keys
    that reduce to the same K'.

    Args:
        inp_dict: Dictionary where keys are binary strings consisting of '0'/'1'.
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
    # --- Basic type checks ---
    if not isinstance(inp_dict, dict):
        raise TypeError("part_dict: 'inp_dict' must be a dict.")
    try:
        indices_tuple: Tuple[int, ...] = tuple(int(i) for i in indices)
    except Exception:
        raise TypeError("part_dict: 'indices' must be an iterable of integers.") from None
    if len(indices_tuple) == 0:
        raise ValueError("part_dict: 'indices' cannot be empty.")

    # --- Validate keys; also determine common bitstring length ---
    lengths = set()
    for k in inp_dict.keys():
        if not isinstance(k, str) or len(k) == 0 or any(c not in "01" for c in k):
            raise ValueError(f"part_dict: invalid binary key {k!r} (expected non-empty '0'/'1' string).")
        lengths.add(len(k))

    if len(lengths) == 0:
        return {}
    if len(lengths) != 1:
        raise ValueError(f"part_dict: all keys must have equal length; got lengths {sorted(lengths)}.")
    L = lengths.pop()

    # --- Normalize and validate indices ---
    # Normalize negatives and check bounds; also forbid duplicates (ambiguous selection)
    normalized_indices: Tuple[int, ...] = tuple(i if i >= 0 else L + i for i in indices_tuple)
    if any(i < 0 or i >= L for i in normalized_indices):
        raise ValueError(f"part_dict: indices out of bounds after normalization for key length {L}: {indices_tuple} → {normalized_indices}.")
    if len(set(normalized_indices)) != len(normalized_indices):
        raise ValueError("part_dict: duplicate indices are not allowed.")

    # --- Build reduced dictionary ---
    new_dict: Dict[str, float] = {}
    for key, value in inp_dict.items():
        # construct substring using MSB=0 positions in the given order
        extracted_bits = "".join(key[i] for i in normalized_indices)
        new_dict[extracted_bits] = new_dict.get(extracted_bits, 0) + value

    return new_dict