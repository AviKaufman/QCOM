import numpy as np
import .progress as ProgressManager
from utils import order_dict, part_dict


"""
Functions for computing classical information measures in a systems. This will be an every growing list of functions.
"""


def compute_shannon_entropy(prob_dict):
    """
    Computes the Shannon entropy of a probability distribution.

    Args:
        prob_dict (dict): A dictionary mapping states to their probabilities.

    Returns:
        float: Shannon entropy.
    """
    total_prob = sum(prob_dict.values())
    entropy = -sum(
        (p / total_prob) * np.log(p / total_prob) for p in prob_dict.values() if p > 0
    )
    return entropy


def compute_reduced_shannon_entropy(prob_dict, configuration, target_region):
    """
    Computes the reduced Shannon entropy for a given region.

    Args:
        prob_dict (dict): A dictionary mapping states to their probabilities.
        configuration (list): A binary list specifying which sites belong to which region (0 for A, 1 for B).
        target_region (int): The target region for entropy calculation (0 for A, 1 for B).

    Returns:
        float: Reduced Shannon entropy.
    """
    # Extract the indices corresponding to the target region (0 for A, 1 for B)
    target_indices = [
        i for i, region in enumerate(configuration) if region == target_region
    ]

    # Obtain the reduced dictionary using the extracted indices
    reduced_dict = part_dict(prob_dict, target_indices)
    reduced_dict = order_dict(reduced_dict)  # Order for consistency

    # Get sorted sequences and corresponding probabilities
    sorted_sequences = list(reduced_dict.keys())
    sorted_probabilities = list(reduced_dict.values())

    # Compute total probability for normalization
    total_prob = sum(sorted_probabilities)

    # Identify unique leftmost parts of the sequences
    unique_leftmost_parts = []
    previous_leftmost = None

    for sequence in sorted_sequences:
        leftmost_part = sequence[: len(target_indices)]  # Extract relevant part
        if leftmost_part != previous_leftmost:
            unique_leftmost_parts.append(leftmost_part)
            previous_leftmost = leftmost_part

    # Initialize reduced probabilities array
    reduced_probabilities = np.zeros(len(unique_leftmost_parts))

    # Sum probabilities for each unique leftmost part
    current_index = -1
    previous_leftmost = None

    for index, sequence in enumerate(sorted_sequences):
        leftmost_part = sequence[: len(target_indices)]
        if leftmost_part != previous_leftmost:
            current_index += 1
            reduced_probabilities[current_index] += sorted_probabilities[index]
            previous_leftmost = leftmost_part
        else:
            reduced_probabilities[current_index] += sorted_probabilities[index]

    # Compute the reduced Shannon entropy
    reduced_entropy = -sum(
        (p / total_prob) * np.log(p / total_prob)
        for p in reduced_probabilities
        if p > 0
    )

    return reduced_entropy


def compute_mutual_information(prob_dict, configuration):
    """
    Computes the classical mutual information between two regions.

    Args:
        prob_dict (dict): A dictionary mapping states to their probabilities.
        configuration (list): A binary list specifying which sites belong to which region (0 for A, 1 for B).

    Returns:
        tuple: (mutual information, Shannon entropy of A, Shannon entropy of B, Shannon entropy of AB)
    """
    # Compute full system entropy
    shan_AB = compute_shannon_entropy(prob_dict)

    # Compute reduced Shannon entropy for region A and B
    shan_A = compute_reduced_shannon_entropy(prob_dict, configuration, target_region=0)
    shan_B = compute_reduced_shannon_entropy(prob_dict, configuration, target_region=1)

    # Compute mutual information
    mutual_information = shan_A + shan_B - shan_AB

    return mutual_information


def cumulative_distribution(binary_dict):
    """
    Compute the cumulative probability distribution from a given binary probability dictionary.

    Args:
        binary_dict (dict): A dictionary where keys are binary strings representing states,
                            and values are their corresponding probabilities.

    Returns:
        tuple: (x_axis, y_axis) representing the cumulative probability distribution.
    """
    # Extract and sort probabilities from the dictionary
    probabilities = np.array(list(binary_dict.values()))
    sorted_probs = np.sort(probabilities)

    # Compute cumulative distribution
    unique_probs, counts = np.unique(sorted_probs, return_counts=True)
    cumulative_prob = np.cumsum(unique_probs * counts)

    # Normalize cumulative probability to ensure it ranges from 0 to 1
    cumulative_prob /= cumulative_prob[-1]

    # Ensure x-axis spans from the smallest probability to 1
    x_axis = np.append(unique_probs, [1])
    y_axis = np.append(cumulative_prob, [1])  # Ensure y-axis ends at 1

    return x_axis, y_axis


def compute_N_of_p_all(probabilities, p_delta=0.1, show_progress=False):
    """
    Efficiently compute N(p) for each unique nonzero probability.

    Args:
        probabilities (array-like): List or array of probabilities.
        p_delta (float): Width in log10 space for the neighborhood.
        show_progress (bool): Whether to show progress updates.

    Returns:
        tuple: (unique_probs, N_values)
    """
    probs = np.array(probabilities)
    probs = probs[probs > 0]
    sorted_probs = np.sort(probs)
    cumulative_probs = np.cumsum(sorted_probs)
    unique_probs = np.unique(sorted_probs)

    def compute_single_N(p):
        log_p = np.log10(p)
        lower = 10 ** (log_p - p_delta / 2)
        upper = 10 ** (log_p + p_delta / 2)

        lower_idx = np.searchsorted(sorted_probs, lower, side="left")
        upper_idx = np.searchsorted(sorted_probs, upper, side="right")

        sigma_lower = cumulative_probs[lower_idx - 1] if lower_idx > 0 else 0.0
        sigma_upper = cumulative_probs[upper_idx - 1] if upper_idx > 0 else 0.0

        return (sigma_upper - sigma_lower) / ((upper - lower) * p)

    N_values = []

    with (
        ProgressManager.progress("Computing N(p)", total_steps=len(unique_probs))
        if show_progress
        else ProgressManager.dummy_context()
    ):
        for i, p in enumerate(unique_probs):
            N_values.append(compute_single_N(p))
            if show_progress:
                ProgressManager.update_progress(i + 1)

    return unique_probs, N_values


def compute_N_of_p(p, sorted_probs, cumulative_probs, p_delta=0.1):
    """
    Compute N(p) at a single value using precomputed arrays.

    Args:
        p (float): Probability to evaluate N(p).
        sorted_probs (np.array): Sorted array of nonzero probabilities.
        cumulative_probs (np.array): Cumulative sum of sorted_probs.
        p_delta (float): Width in log10 space.

    Returns:
        float: N(p)
    """
    if p <= 0:
        return 0.0

    log_p = np.log10(p)
    lower = 10 ** (log_p - p_delta / 2)
    upper = 10 ** (log_p + p_delta / 2)

    lower_idx = np.searchsorted(sorted_probs, lower, side="left")
    upper_idx = np.searchsorted(sorted_probs, upper, side="right")

    sigma_lower = cumulative_probs[lower_idx - 1] if lower_idx > 0 else 0.0
    sigma_upper = cumulative_probs[upper_idx - 1] if upper_idx > 0 else 0.0

    return (sigma_upper - sigma_lower) / ((upper - lower) * p)
