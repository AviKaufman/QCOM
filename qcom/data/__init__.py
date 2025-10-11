# qcom/data/__init__.py
"""
QCOM Data Utilities

This subpackage provides tools for working with raw quantum measurement
data (bit-string → counts), probabilities, and noise models.

Modules
-------
- noise      : Simulated noise models (currently readout error injection).
- ops        : Basic operations on probability/count dictionaries
               (normalization, truncation, pretty-printing).
- sampling   : Sampling and dataset-combination utilities.

Typical usage
-------------
>>> from qcom.data import normalize_to_probabilities, sample_data, introduce_error
>>> counts = {"00": 50, "01": 25, "10": 25}
>>> probs = normalize_to_probabilities(counts, total_count=100)
>>> noisy = introduce_error(counts, ground_rate=0.02, excited_rate=0.1)
>>> sampled = sample_data(noisy, total_count=100, sample_size=1000)

Notes
-----
This namespace re-exports the most commonly used functions. For more
specialized utilities, import directly from the submodules.
"""

from .noise import introduce_error
from .ops import normalize_to_probabilities, truncate_probabilities, print_most_probable_data
from .sampling import sample_data, combine_datasets

__all__ = [
    "introduce_error",
    "normalize_to_probabilities",
    "truncate_probabilities",
    "print_most_probable_data",
    "sample_data",
    "combine_datasets",
]