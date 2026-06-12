"""
QCOM Data Utilities

This subpackage provides tools for working with raw quantum measurement
data (bitstring -> counts), probabilities, and noise models.

Modules
-------
- noise      : Simulated noise models (currently readout error injection).
- ops        : Basic operations on probability/count dictionaries
               (normalization, truncation, pretty-printing).
- sampling   : Sampling and dataset-combination utilities.

Typical usage
-------------
>>> from qcom.data import normalize_to_probabilities, sample_counts, apply_readout_error
>>> counts = {"00": 50, "01": 25, "10": 25}
>>> probs = normalize_to_probabilities(counts, total_count=100)
>>> noisy = apply_readout_error(counts, ground_rate=0.02, excited_rate=0.1)
>>> sampled = sample_counts(noisy, total_count=100, sample_size=1000)

Notes
-----
This namespace re-exports the most commonly used functions. For more
specialized utilities, import directly from the submodules.
"""

from qcom.core import CountsData, ProbabilityData

from .noise import apply_readout_error, introduce_error
from .ops import (
    normalize_to_probabilities,
    truncate_probabilities,
    print_most_probable_bitstrings,
    print_most_probable_data,
)
from .sampling import sample_counts, combine_bitstring_datasets, sample_data, combine_datasets

__all__ = [
    "CountsData",
    "ProbabilityData",
    "apply_readout_error",
    "introduce_error",
    "normalize_to_probabilities",
    "truncate_probabilities",
    "print_most_probable_bitstrings",
    "print_most_probable_data",
    "sample_counts",
    "sample_data",
    "combine_bitstring_datasets",
    "combine_datasets",
]
