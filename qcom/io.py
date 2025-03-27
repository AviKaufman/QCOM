import random
import os
from .progress import ProgressManager
import pandas as pd

"""
I/O functions for loading and saving external data files
"""


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
