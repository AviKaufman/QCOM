import numpy as np
from .progress import ProgressManager
from .utils import order_dict, part_dict


"""
Functions for computing classical information measures in a systems. This will be an every growing list of functions.
"""


def compute_shannon_entropy(prob_dict, total_prob=1):
    """
    Computes the Shannon entropy of a probability distribution.

    Args:
        prob_dict (dict): A dictionary mapping states to their probabilities.

    Returns:
        float: Shannon entropy.
    """

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


def compute_mutual_information(prob_dict, configuration, total_count=1):
    """
    Computes the classical mutual information between two regions.

    Args:
        prob_dict (dict): A dictionary mapping states to their probabilities.
        configuration (list): A binary list specifying which sites belong to which region (0 for A, 1 for B).

    Returns:
        tuple: (mutual information, Shannon entropy of A, Shannon entropy of B, Shannon entropy of AB)
    """
    # Compute full system entropy
    shan_AB = compute_shannon_entropy(prob_dict, total_count)

    # Compute reduced Shannon entropy for region A and B
    shan_A = compute_reduced_shannon_entropy(prob_dict, configuration, target_region=0)
    shan_B = compute_reduced_shannon_entropy(prob_dict, configuration, target_region=1)

    # Compute mutual information
    mutual_information = shan_A + shan_B - shan_AB

    return mutual_information


def compute_conditional_entropy(prob_dict, configuration, total_count=1):
    """
    Computes the conditional entropy of region A given region B.

    Args:
        prob_dict (dict): A dictionary mapping states to their probabilities.
        configuration (list): A binary list specifying which sites belong to which region (0 for A, 1 for B).

    Returns:
        float: Conditional entropy of region A given region B.
    """
    # Compute full system entropy
    shan_AB = compute_shannon_entropy(prob_dict, total_count)
    # Compute reduced Shannon entropy for region B
    shan_B = compute_reduced_shannon_entropy(prob_dict, configuration, target_region=1)

    # Compute conditional entropy
    conditional_entropy = shan_AB - shan_B
    return conditional_entropy


def cumulative_probability_at_value(binary_dict, value):
    """
    Compute the cumulative sum of probabilities in binary_dict that are less than or equal to the given value.

    Args:
        binary_dict (dict): A dictionary where keys are binary strings representing states,
                            and values are their corresponding probabilities.
        value (float): The threshold probability.

    Returns:
        float: The cumulative sum of probabilities (i.e., the sum of all probabilities <= value).
    """
    # Convert the dictionary values to a numpy array for easy filtering
    probabilities = np.array(list(binary_dict.values()))

    # Filter and sum all probabilities that are less than or equal to 'value'
    cumulative_sum = np.sum(probabilities[probabilities <= value])

    return cumulative_sum


def cumulative_distribution(binary_dict, bins=None, p_max=1.0):
    """
    Compute a cumulative-distribution “step” from a dict of {state_str → probability}.
    - If bins=None:  use the old behavior (unique probs + append (1,1)).
    - If bins=N:     produce N log-spaced bin centers between p_min and 1.0,
                     evaluate CDF at each center, normalize, then append (1,1).

    Args:
        binary_dict (dict[str, float]):  keys are bit-strings, values must sum to 1.0.
        bins (int or None):  if None, return the “unique-prob” CDF; else must be a
                             positive integer N, returning an N+1-entry CDF on a fixed grid.

    Returns:
        (x_axis, y_axis) as two 1D numpy arrays of length
        - bins=None:  len = (# unique probs) + 1
        - bins=N:     len = N + 1
    """
    if not binary_dict:
        raise ValueError("cumulative_distribution: input dict is empty")

    # 1) Turn values into a NumPy array of probabilities
    probs = np.array(list(binary_dict.values()), dtype=float)
    # check that they sum to 1.0
    if not np.isclose(np.sum(probs), 1.0):
        raise ValueError("cumulative_distribution: input dict values must sum to 1.0")

    if bins is None:
        sorted_p = np.sort(probs)
        unique_p, counts = np.unique(sorted_p, return_counts=True)
        cumulative = np.cumsum(unique_p * counts, dtype=float)
        cumulative /= cumulative[-1]  # ensure final = 1.0

        x_axis = np.append(unique_p, 1.0)
        y_axis = np.append(cumulative, 1.0)
        return x_axis, y_axis

    N = int(bins)
    if N < 2:
        raise ValueError(
            "cumulative_distribution: if 'bins' is specified, it must be an integer ≥ 2"
        )

    # 1) Find the smallest nonzero probability
    positive = probs[probs > 0.0]
    if positive.size == 0:
        # If for some reason all entries are zero, choose a fallback p_min
        p_min = 1.0 / N
    else:
        p_min = positive.min()

    if p_max:
        # find p_min in log10 space
        log_p_max = np.log10(p_max)
    else:
        # If p_max is not specified, use 1.0 as the maximum probability
        log_p_max = 0.0

    # 2) Make exactly N log‐spaced points from p_min to 1.0 (inclusive)
    #    np.logspace(log10(p_min), 0.0, num=N) gives N values, where
    #      index 0 = 10^log10(p_min) = p_min,
    #      index N−1 = 10^0 = 1.0.
    x_axis = np.logspace(np.log10(p_min), log_p_max, num=N)

    # 3) Compute CDF at each x_axis[i] = sum of all p ≤ x_axis[i].
    sorted_p = np.sort(probs)
    n = sorted_p.size
    idx = 0
    running_sum = 0.0

    y_axis = np.zeros(N, dtype=float)
    for i, c in enumerate(x_axis):
        while idx < n and sorted_p[idx] <= c:
            running_sum += sorted_p[idx]
            idx += 1
        y_axis[i] = running_sum

    # 4) Normalize so that y_axis[-1] == 1.0 (if running_sum > 0)
    if running_sum > 0.0:
        y_axis /= running_sum
    else:
        # All probs were zero—leave y_axis as zeros (though this is unlikely)
        pass

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

    # Check if probabilites is a dictionary
    if isinstance(probabilities, dict):
        # Extract values from the dictionary
        probabilities = list(probabilities.values())

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
