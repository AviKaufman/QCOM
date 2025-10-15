# qcom/data/sampling.py
"""
Sampling and dataset combination utilities for QCOM.

Goal
----
Provide tools to resample quantum bitstring distributions and to merge
datasets (counts or probabilities) in a principled way.

Scope (current)
---------------
• `sample_data`: Generate synthetic datasets by sampling from a probability distribution.
• `combine_datasets`: Merge two datasets safely, handling both probability and count forms.

Design notes
------------
• Built for compatibility with QCOM’s normalization utilities.
• Progress reporting integrated with `ProgressManager` (optional).
• Error-guarded: prevents mixing of probability and count datasets.

Typical usage
-------------
>>> from qcom.data.sampling import sample_data, combine_datasets
>>> from qcom.data.ops import normalize_to_probabilities
>>> counts = {"00": 50, "01": 25, "10": 25}
>>> probs = normalize_to_probabilities(counts, total_count=100)
>>> sampled = sample_data(counts, total_count=100, sample_size=1000, show_progress=True)
>>> merged = combine_datasets(probs, probs, show_progress=True)

Future extensions (non-breaking)
--------------------------------
• Stratified sampling or weighted re-sampling.
• Bootstrapping helpers for error estimation.
• Dataset splitting (train/test partitioning).
"""

import random
from typing import Dict, Union

from .._internal import ProgressManager
from ..data.ops import normalize_to_probabilities


# ========================================== Sampling Utilities ==========================================

def sample_data(
    data: Dict[str, int],
    total_count: int,
    sample_size: int,
    update_interval: int = 100,
    show_progress: bool = False,
) -> Dict[str, int]:
    """
    Sample bit strings based on their probabilities.

    Parameters
    ----------
    data : dict[str, int]
        Dictionary mapping bit strings to raw counts.
    total_count : int
        Total number of counts (used for normalization).
    sample_size : int
        Number of samples to generate.
    update_interval : int, optional
        Frequency of progress updates (default = 100).
    show_progress : bool, optional
        Whether to display progress updates.

    Returns
    -------
    dict[str, int]
        Dictionary mapping sampled bit strings to their new counts.
    """
    # -------------------- Normalize input data --------------------
    normalized_data = normalize_to_probabilities(data, total_count)
    sequences = list(normalized_data.keys())
    probabilities = list(normalized_data.values())

    sampled_dict: Dict[str, int] = {}

    # -------------------- Perform sampling with optional progress --------------------
    with (
        ProgressManager.progress("Sampling data", total_steps=sample_size)
        if show_progress
        else ProgressManager.dummy_context()
    ):
        sampled_sequences = random.choices(
            sequences, weights=probabilities, k=sample_size
        )

        for idx, sequence in enumerate(sampled_sequences):
            sampled_dict[sequence] = sampled_dict.get(sequence, 0) + 1
            if show_progress and idx % update_interval == 0:
                ProgressManager.update_progress(idx + 1)

        if show_progress:
            ProgressManager.update_progress(sample_size)  # Ensure 100% completion

    return sampled_dict


# ========================================== Dataset Combination ==========================================

def combine_datasets(
    data1: Dict[str, Union[int, float]],
    data2: Dict[str, Union[int, float]],
    tol: float = 1e-6,
    update_interval: int = 100,
    show_progress: bool = False,
) -> Dict[str, Union[int, float]]:
    """
    Combine two datasets (counts or probabilities).

    Rules
    -----
    • If both datasets are probabilities (sum ≈ 1), merge and renormalize.
    • If both datasets are counts, merge counts directly.
    • If one dataset is probabilities and the other is counts, raise an error.

    Parameters
    ----------
    data1, data2 : dict[str, int or float]
        Datasets to combine.
    tol : float, optional
        Tolerance for checking probability normalization (default = 1e-6).
    update_interval : int, optional
        Frequency of progress updates.
    show_progress : bool, optional
        Whether to display progress updates.

    Returns
    -------
    dict[str, int or float]
        The combined dataset. Normalized if both inputs were probabilities.

    Raises
    ------
    ValueError
        If one dataset is probabilities and the other is counts.
    """
    # -------------------- Detect type of datasets --------------------
    total1 = sum(data1.values())
    total2 = sum(data2.values())

    is_prob1 = abs(total1 - 1.0) < tol
    is_prob2 = abs(total2 - 1.0) < tol

    if is_prob1 and is_prob2:
        data_type = "probabilities"
    elif (is_prob1 and not is_prob2) or (not is_prob1 and is_prob2):
        raise ValueError(
            "Cannot combine a dataset of probabilities with a dataset of counts. "
            "Convert one to the other before combining."
        )
    else:
        data_type = "counts"

    # -------------------- Merge datasets with optional progress --------------------
    combined: Dict[str, Union[int, float]] = {}
    all_keys = set(data1.keys()).union(data2.keys())
    total_keys = len(all_keys)

    with (
        ProgressManager.progress("Combining datasets", total_steps=total_keys)
        if show_progress
        else ProgressManager.dummy_context()
    ):
        for idx, key in enumerate(all_keys):
            combined[key] = data1.get(key, 0) + data2.get(key, 0)

            if show_progress and idx % update_interval == 0:
                ProgressManager.update_progress(idx + 1)

        if show_progress:
            ProgressManager.update_progress(total_keys)

    # -------------------- Renormalize if probability data --------------------
    if data_type == "probabilities":
        combined_total = sum(combined.values())
        combined = {key: value / combined_total for key, value in combined.items()}

    return combined