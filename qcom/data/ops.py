# qcom/data/ops.py
"""
Basic dictionary operations for QCOM probability data.

Goal
----
Provide lightweight, composable utilities for transforming and inspecting
probability distributions derived from quantum bitstring data.

Scope (current)
---------------
• Normalization of raw counts into probability dictionaries.
• Truncation of low-probability entries without renormalization.
• Formatted printing of the most probable bit strings.

Design notes
------------
• Keep stateless, pure functions: input → output, no hidden side effects.
• Avoid heavy dependencies — all functions are standard Python only.
• Functions assume input is a dictionary mapping bitstrings → floats.

Typical usage
-------------
>>> from qcom.data.ops import normalize_to_probabilities, truncate_probabilities
>>> counts = {"00": 50, "01": 25, "10": 25, "11": 0}
>>> probs = normalize_to_probabilities(counts, total_count=100)
>>> truncated = truncate_probabilities(probs, threshold=0.2)
>>> print_most_probable_data(probs, n=2)

Future extensions (non-breaking)
--------------------------------
• Optional renormalization after truncation.
• Sorting helpers (by Hamming weight, lexicographic order, etc.).
• Export utilities (CSV/JSON pretty-print).
"""

from typing import Dict


# ========================================== Probability Operations ==========================================

def normalize_to_probabilities(data: Dict[str, int], total_count: int) -> Dict[str, float]:
    """
    Convert raw counts to probabilities.

    Args:
        data (dict): Dictionary mapping bit strings to counts (int).
        total_count (int): Total number of counts across all entries.

    Returns:
        dict: Dictionary mapping bit strings to probabilities.

    Raises:
        ValueError: If total_count is None or zero.
    """
    # -------------------- Validation --------------------
    if total_count is None:
        raise ValueError("Total count must be provided for normalization.")
    if total_count == 0:
        raise ValueError("Total count is zero; cannot normalize to probabilities.")

    # -------------------- Core Computation --------------------
    return {key: value / total_count for key, value in data.items()}


# ========================================== Truncation Operations ==========================================

def truncate_probabilities(input_dict: Dict[str, float], threshold: float) -> Dict[str, float]:
    """
    Remove entries below a probability threshold (no renormalization).

    Args:
        input_dict (dict): Dictionary mapping bit strings to probabilities.
        threshold (float): Minimum probability to keep an entry.

    Returns:
        dict: Filtered dictionary with only entries >= threshold.
    """
    # -------------------- Core Computation --------------------
    return {
        bitstring: prob
        for bitstring, prob in input_dict.items()
        if prob >= threshold
    }


# ========================================== Output / Reporting ==========================================

def print_most_probable_data(normalized_data: Dict[str, float], n: int = 10) -> None:
    """
    Print the `n` most probable bit strings in descending order.

    Args:
        normalized_data (dict): Dictionary mapping bit strings to probabilities.
        n (int): Number of top entries to print.
    """
    # -------------------- Sorting --------------------
    sorted_data = sorted(normalized_data.items(), key=lambda x: x[1], reverse=True)

    print(f"Top {n} Most probable bit strings:")

    # Determine index width for aligned output (e.g., " 1.", "10.")
    max_index_width = len(str(n))

    # -------------------- Printing --------------------
    for idx, (sequence, probability) in enumerate(sorted_data[:n], start=1):
        print(
            f"{str(idx).rjust(max_index_width)}.  "
            f"Bit string: {sequence}, Probability: {probability:.8f}"
        )