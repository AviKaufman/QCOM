from progress_manager import ProgressManager
import random
import os
import pandas as pd


def parse_file(
    file_path, sample_size=None, update_interval=500000, show_progress=False
):
    """
    Parse the file and optionally sample data while reading.

    This version streams the file line by line and updates progress only every
    update_interval lines based on the file's byte size.

    Args:
        file_path (str): Path to the input file.
        sample_size (int, optional): Number of samples to retain (None means full processing).
        update_interval (int, optional): Number of lines before updating progress.
        show_progress (bool, optional): Whether to display progress updates.

    Returns:
        data (dict): A dictionary mapping binary sequences to their raw counts.
        total_count (float): The sum of counts across all sequences.
    """
    data = {}
    total_count = 0.0
    valid_lines = 0

    file_size = os.path.getsize(file_path)
    bytes_read = 0

    with open(file_path, "r") as file:
        with (
            ProgressManager.progress("Parsing file", total_steps=file_size)
            if show_progress
            else ProgressManager.dummy_context()
        ):
            for idx, line in enumerate(file):
                bytes_read += len(line)
                if show_progress and idx % update_interval == 0:
                    ProgressManager.update_progress(bytes_read)
                try:
                    line = line.strip()
                    if not line:
                        continue

                    binary_sequence, count_str = line.split()
                    count = float(count_str)
                    total_count += count
                    valid_lines += 1

                    if sample_size and len(data) < sample_size:
                        data[binary_sequence] = count
                    elif sample_size:
                        # Reservoir sampling using the count of valid lines.
                        replace_idx = random.randint(0, valid_lines - 1)
                        if replace_idx < sample_size:
                            keys = list(data.keys())
                            data[keys[replace_idx]] = count
                    else:
                        data[binary_sequence] = count

                except Exception as e:
                    print(f"Error reading line '{line}' in {file_path}: {e}")

            if show_progress:
                ProgressManager.update_progress(file_size)

    return data, total_count


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


def save_data(data, savefile, update_interval=100, show_progress=False):
    """
    Save the data to a file using the same convention as in parse_file, with optional progress tracking.

    Each line in the file will contain:
        <state> <value>
    where 'state' is the binary sequence and 'value' is the associated count or probability.

    Args:
        data (dict): Dictionary with keys as states and values as counts or probabilities.
        savefile (str): The path to the file where the data will be saved.
        update_interval (int, optional): Frequency at which progress updates occur.
        show_progress (bool, optional): Whether to display progress updates.
    """
    states = list(data.keys())
    total_states = len(states)

    with open(savefile, "w") as f:
        with (
            ProgressManager.progress("Saving data", total_steps=total_states)
            if show_progress
            else ProgressManager.dummy_context()
        ):
            for idx, state in enumerate(states):
                f.write(f"{state} {data[state]}\n")

                if show_progress and idx % update_interval == 0:
                    ProgressManager.update_progress(idx + 1)

            if show_progress:
                ProgressManager.update_progress(total_states)  # Ensure 100% completion


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


def save_dict_to_parquet(data_dict, file_name):
    """
    Saves a dictionary of key-value pairs (e.g., {"state": prob}) to a Parquet file.

    Parameters:
        data_dict (dict): A dictionary where keys are states and values are probabilities.
        file_name (str): The name of the Parquet file to save.
    """
    total_steps = 3
    with ProgressManager.progress(
        "Saving dictionary to Parquet", total_steps=total_steps
    ):
        # Step 1: Convert dictionary to a list of items.
        items = list(data_dict.items())
        ProgressManager.update_progress(1)

        # Step 2: Create a DataFrame from the items.
        df = pd.DataFrame(items, columns=["state", "probability"])
        ProgressManager.update_progress(2)

        # Step 3: Save the DataFrame as a Parquet file.
        df.to_parquet(file_name, engine="pyarrow", index=False)
        ProgressManager.update_progress(3)

    print(f"Dictionary saved to {file_name}")


def parse_parq(file_name):
    """
    Reads a Parquet file and converts it back into a dictionary.

    Parameters:
        file_name (str): The Parquet file name to read.

    Returns:
        dict: A dictionary where keys are states and values are probabilities.
    """
    total_steps = 2
    with ProgressManager.progress("Parsing Parquet file", total_steps=total_steps):
        # Step 1: Read the Parquet file into a DataFrame.
        df = pd.read_parquet(file_name, engine="pyarrow")
        ProgressManager.update_progress(1)

        # Step 2: Convert the DataFrame into a dictionary.
        data_dict = dict(zip(df["state"], df["probability"]))
        ProgressManager.update_progress(2)

    return data_dict


# Test code
if __name__ == "__main__":
    file_path = "../EntanglementCalculation/1_billion_shots/Rba_2.0/19_rungs.txt"

    # Process without sampling
    processed_data, total_count = parse_file(file_path, sample_size=None)
    print_most_probable_data(processed_data, n=10)

    # Process with sampling
    sampled_data = sample_data(processed_data, total_count, sample_size=100)
    print(f"Sampled data: {sampled_data}")
