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
"""


# -------------------- Order dictionary by integer value --------------------

def order_dict(inp_dict):
    """
    Orders a dictionary based on binary keys interpreted as integers.

    Args:
        inp_dict (dict): Dictionary where keys are binary strings.

    Returns:
        dict: Ordered dictionary sorted by integer values of binary keys.
    """
    ordered_items = sorted(inp_dict.items(), key=lambda item: int(item[0], 2))
    return dict(ordered_items)


# -------------------- Extract subset of bits --------------------

def part_dict(inp_dict, indices):
    """
    Extracts a subset of bits from each binary string based on given indices.

    Args:
        inp_dict (dict): Dictionary where keys are binary strings.
        indices (list): List of indices specifying which bits to extract.

    Returns:
        dict: New dictionary where keys contain only the extracted bits.
              Values are summed if multiple original keys reduce to the same substring.
    """
    new_dict = {}
    for key, value in inp_dict.items():
        extracted_bits = "".join(key[i] for i in indices)
        if extracted_bits in new_dict:
            new_dict[extracted_bits] += value
        else:
            new_dict[extracted_bits] = value
    return new_dict