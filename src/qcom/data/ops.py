"""
Basic dictionary operations for QCOM probability data.
"""

from typing import Dict

from qcom._internal.deprecations import warn_deprecated_alias
from qcom.core import CountsData, ProbabilityData

__all__ = [
    "normalize_to_probabilities",
    "truncate_probabilities",
    "print_most_probable_bitstrings",
    "print_most_probable_data",
]


def normalize_to_probabilities(
    data: Dict[str, int] | CountsData,
    total_count: int | None = None,
    *,
    return_data: bool = False,
) -> Dict[str, float] | ProbabilityData:
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
    source = None
    if isinstance(data, CountsData):
        counts = data.to_dict()
        total_count = data.shots if total_count is None else total_count
        source = data.source
    else:
        counts = data

    if total_count is None:
        raise ValueError("Total count must be provided for normalization.")
    if total_count == 0:
        raise ValueError("Total count is zero; cannot normalize to probabilities.")

    probabilities = {key: value / total_count for key, value in counts.items()}
    if return_data:
        return ProbabilityData(probabilities, source=source)
    return probabilities


def truncate_probabilities(input_dict: Dict[str, float], threshold: float) -> Dict[str, float]:
    """
    Remove entries below a probability threshold (no renormalization).

    Args:
        input_dict (dict): Dictionary mapping bit strings to probabilities.
        threshold (float): Minimum probability to keep an entry.

    Returns:
        dict: Filtered dictionary with only entries >= threshold.
    """
    return {bitstring: prob for bitstring, prob in input_dict.items() if prob >= threshold}


def print_most_probable_bitstrings(probabilities: Dict[str, float], n: int = 10) -> None:
    """
    Print the `n` most probable bit strings in descending order.

    Args:
        probabilities (dict): Dictionary mapping bit strings to probabilities.
        n (int): Number of top entries to print.
    """
    sorted_probabilities = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)

    print(f"Top {n} Most probable bit strings:")

    max_index_width = len(str(n))

    for index, (bitstring, probability) in enumerate(sorted_probabilities[:n], start=1):
        print(
            f"{str(index).rjust(max_index_width)}.  "
            f"Bit string: {bitstring}, Probability: {probability:.8f}"
        )


def print_most_probable_data(normalized_data: Dict[str, float], n: int = 10) -> None:
    """Deprecated compatibility alias for `print_most_probable_bitstrings`."""
    warn_deprecated_alias("print_most_probable_data", "print_most_probable_bitstrings")
    print_most_probable_bitstrings(normalized_data, n=n)
