import progress as ProgressManager
import random

"""
This modules provides functions for manipulating and analyzing data.
"""


def normalize_to_probabilities(data, total_count):
    """
    Convert raw counts to probabilities.

    Returns:
        normalized_data (dict): A dictionary with probabilities.
    """
    if total_count == 0:
        raise ValueError("Total count is zero; cannot normalize to probabilities.")
    normalized_data = {key: value / total_count for key, value in data.items()}
    return normalized_data


def sample_data(
    data, total_count, sample_size, update_interval=100, show_progress=False
):
    """
    Sample bit strings based on their probabilities.

    Args:
        data (dict): Dictionary of raw counts.
        total_count (float): Sum of all counts (used for normalization).
        sample_size (int): Number of samples to generate.
        update_interval (int, optional): Number of samples before updating progress.
        show_progress (bool, optional): Whether to display progress updates.

    Returns:
        dict: A dictionary mapping sampled bit strings to their probabilities.
    """
    normalized_data = normalize_to_probabilities(data, total_count)
    sequences = list(normalized_data.keys())
    probabilities = list(normalized_data.values())

    sampled_dict = {}

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

    total_sampled_count = sum(sampled_dict.values())
    return {key: count / total_sampled_count for key, count in sampled_dict.items()}


def introduce_error_data(
    data,
    total_count,
    ground_rate=0.01,
    excited_rate=0.08,
    update_interval=100,
    show_progress=False,
):
    """
    Introduce bit-flipping errors to the dataset with separate error rates for ground and excited states.

    If a bit is '1', it has an 'excited_rate' chance of being flipped to '0'.
    Conversely, if a bit is '0', it has a 'ground_rate' chance of being flipped to '1'.

    Args:
        data (dict): Dictionary of raw counts.
        total_count (float): Sum of all counts (used for normalization).
        ground_rate (float, optional): Probability of a '0' flipping to '1'. Default is 0.01.
        excited_rate (float, optional): Probability of a '1' flipping to '0'. Default is 0.08.
        update_interval (int, optional): Number of sequences before updating progress.
        show_progress (bool, optional): Whether to display progress updates.

    Returns:
        dict: A dictionary with probabilities after errors are introduced.
    """
    print("Introducing errors to the data...")
    normalized_data = normalize_to_probabilities(data, total_count)
    new_data = {}
    sequences = list(normalized_data.keys())

    with (
        ProgressManager.progress("Introducing errors", total_steps=len(sequences))
        if show_progress
        else ProgressManager.dummy_context()
    ):
        for idx, sequence in enumerate(sequences):
            modified_sequence = list(sequence)

            for i in range(len(modified_sequence)):
                if modified_sequence[i] == "1" and random.random() < excited_rate:
                    modified_sequence[i] = "0"
                elif modified_sequence[i] == "0" and random.random() < ground_rate:
                    modified_sequence[i] = "1"

            new_sequence = "".join(modified_sequence)
            new_data[new_sequence] = new_data.get(new_sequence, 0) + 1

            if show_progress and idx % update_interval == 0:
                ProgressManager.update_progress(idx + 1)

        if show_progress:
            ProgressManager.update_progress(len(sequences))  # Ensure 100% completion

    total_new_count = sum(new_data.values())
    return {key: count / total_new_count for key, count in new_data.items()}


def print_most_probable_data(normalized_data, n=10):
    """
    Print the n most probable bit strings with evenly spaced formatting.
    """
    sorted_data = sorted(normalized_data.items(), key=lambda x: x[1], reverse=True)
    print(f"Most probable {n} bit strings:")

    # Find max index width (for up to 99, this is 2)
    max_index_width = len(str(n))

    for idx, (sequence, probability) in enumerate(sorted_data[:n], start=1):
        print(
            f"{str(idx).rjust(max_index_width)}.  Bit string: {sequence}, Probability: {probability:.8f}"
        )


def combine_datasets(data1, data2, tol=1e-6, update_interval=100, show_progress=False):
    """
    Combine two datasets (dictionaries mapping states to counts or probabilities).

    If both datasets are probabilities (sum ≈ 1), combine and renormalize the result so that it sums to 1.
    If both datasets are counts (i.e. neither sums to 1), simply combine the counts without normalization.
    If one dataset is probabilities and the other is counts, raise an error.

    Args:
        data1, data2 (dict): The datasets to combine.
        tol (float, optional): Tolerance for checking if a dataset is probabilities (default is 1e-6).
        update_interval (int, optional): Frequency at which progress updates occur.
        show_progress (bool, optional): Whether to display progress updates.

    Returns:
        combined (dict): The combined dataset.
            - If both inputs are probabilities, the returned dataset is normalized.
            - If both inputs are counts, the returned dataset is not normalized.

    Raises:
        ValueError: If one dataset is probabilities and the other is counts.
    """
    total1 = sum(data1.values())
    total2 = sum(data2.values())

    is_prob1 = abs(total1 - 1.0) < tol
    is_prob2 = abs(total2 - 1.0) < tol

    if is_prob1 and is_prob2:
        data_type = "probabilities"
    elif (is_prob1 and not is_prob2) or (not is_prob1 and is_prob2):
        raise ValueError(
            "Cannot combine a dataset of probabilities with a dataset of counts. "
            "Please convert one to the other before combining."
        )
    else:
        data_type = "counts"

    combined = {}
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
            ProgressManager.update_progress(total_keys)  # Ensure 100% completion

    if data_type == "probabilities":
        combined_total = sum(combined.values())
        combined = {key: value / combined_total for key, value in combined.items()}

    return combined
