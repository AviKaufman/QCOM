import numpy as np


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


def part_dict(inp_dict, indices):
    """
    Extracts a subset of bits from each binary string based on given indices.

    Args:
        inp_dict (dict): Dictionary where keys are binary strings.
        indices (list): List of indices specifying which bits to extract.

    Returns:
        dict: New dictionary where keys contain only the extracted bits.
    """
    new_dict = {}

    for key, value in inp_dict.items():
        extracted_bits = "".join(
            key[i] for i in indices
        )  # Extract only relevant indices
        if extracted_bits in new_dict:
            new_dict[extracted_bits] += value  # Sum probabilities for duplicates
        else:
            new_dict[extracted_bits] = value

    return new_dict


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
