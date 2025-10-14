# qcom/metrics/classical.py
"""
qcom.metrics.classical
======================

Classical information-theoretic measures on probability distributions.

Purpose
-------
Provide basic Shannon-type metrics from classical information theory,
computed directly from measurement probability dictionaries.

Functions
---------
- compute_shannon_entropy(prob_dict, total_prob=1):
    Shannon entropy of a probability distribution.

- compute_reduced_shannon_entropy(prob_dict, configuration, target_region):
    Reduced Shannon entropy of a subsystem defined by a site configuration.

- compute_mutual_information(prob_dict, configuration, total_count=1):
    Classical mutual information between two regions.

- compute_conditional_entropy(prob_dict, configuration, total_count=1):
    Conditional entropy of one region given the other.
"""

import numpy as np
from .bitstrings import order_dict, part_dict


# -------------------- Shannon Entropy --------------------

def compute_shannon_entropy(prob_dict, total_prob=1):
    """
    Computes the Shannon entropy of a probability distribution.

    Args:
        prob_dict (dict): A dictionary mapping states to their probabilities.
        total_prob (float, optional): Normalization constant. Default = 1.

    Returns:
        float: Shannon entropy.
    """
    entropy = -sum(
        (p / total_prob) * np.log(p / total_prob)
        for p in prob_dict.values()
        if p > 0
    )
    return entropy


# -------------------- Reduced Shannon Entropy --------------------

def compute_reduced_shannon_entropy(prob_dict, configuration, target_region):
    """
    Computes the reduced Shannon entropy for a given region.

    Args:
        prob_dict (dict): A dictionary mapping states to their probabilities.
        configuration (list): A binary list specifying which sites belong to
            which region (0 for A, 1 for B).
        target_region (int): Region to compute entropy for (0 = A, 1 = B).

    Returns:
        float: Reduced Shannon entropy for the specified region.
    """
    # Extract target indices
    target_indices = [
        i for i, region in enumerate(configuration) if region == target_region
    ]

    # Reduce dictionary to target indices
    reduced_dict = part_dict(prob_dict, target_indices)
    reduced_dict = order_dict(reduced_dict)

    sorted_sequences = list(reduced_dict.keys())
    sorted_probabilities = list(reduced_dict.values())
    total_prob = sum(sorted_probabilities)

    # Identify unique leftmost substrings
    unique_leftmost_parts = []
    previous_leftmost = None
    for sequence in sorted_sequences:
        leftmost_part = sequence[: len(target_indices)]
        if leftmost_part != previous_leftmost:
            unique_leftmost_parts.append(leftmost_part)
            previous_leftmost = leftmost_part

    # Accumulate probabilities for each unique substring
    reduced_probabilities = np.zeros(len(unique_leftmost_parts))
    current_index = -1
    previous_leftmost = None
    for idx, sequence in enumerate(sorted_sequences):
        leftmost_part = sequence[: len(target_indices)]
        if leftmost_part != previous_leftmost:
            current_index += 1
            previous_leftmost = leftmost_part
        reduced_probabilities[current_index] += sorted_probabilities[idx]

    # Compute entropy
    reduced_entropy = -sum(
        (p / total_prob) * np.log(p / total_prob)
        for p in reduced_probabilities
        if p > 0
    )
    return reduced_entropy


# -------------------- Mutual Information --------------------

def compute_mutual_information(prob_dict, configuration, total_count=1):
    """
    Computes the classical mutual information between two regions.

    Args:
        prob_dict (dict): A dictionary mapping states to their probabilities.
        configuration (list): A binary list specifying which sites belong to
            which region (0 for A, 1 for B).
        total_count (float, optional): Normalization constant. Default = 1.

    Returns:
        tuple: (mutual_information, Shannon entropy of A,
                Shannon entropy of B, Shannon entropy of AB)
    """
    shan_AB = compute_shannon_entropy(prob_dict, total_count)
    shan_A = compute_reduced_shannon_entropy(prob_dict, configuration, target_region=0)
    shan_B = compute_reduced_shannon_entropy(prob_dict, configuration, target_region=1)

    mutual_information = shan_A + shan_B - shan_AB
    return mutual_information, shan_A, shan_B, shan_AB


# -------------------- Conditional Entropy --------------------

def compute_conditional_entropy(prob_dict, configuration, total_count=1):
    """
    Computes the conditional entropy of region A given region B.

    Args:
        prob_dict (dict): A dictionary mapping states to their probabilities.
        configuration (list): A binary list specifying which sites belong to
            which region (0 for A, 1 for B).
        total_count (float, optional): Normalization constant. Default = 1.

    Returns:
        float: Conditional entropy H(A|B).
    """
    shan_AB = compute_shannon_entropy(prob_dict, total_count)
    shan_B = compute_reduced_shannon_entropy(prob_dict, configuration, target_region=1)
    return shan_AB - shan_B