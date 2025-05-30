import random
import os
from .progress import ProgressManager
import pandas as pd
import json

"""
I/O functions for loading and saving external data files
"""


def parse_file(file_path, update_interval=500000, show_progress=False):
    """
    Parse the file and return counts for every binary sequence.

    Args:
        file_path (str): Path to the input file.
        update_interval (int, optional): Number of lines before updating progress.
        show_progress (bool, optional): Whether to display progress updates.

    Returns:
        data (dict): A dictionary mapping binary sequences to their raw counts.
        total_count (float): The sum of counts across all sequences.
    """
    data = {}
    total_count = 0.0

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

                line = line.strip()
                if not line:
                    continue

                try:
                    binary_sequence, count_str = line.split()
                    count = float(count_str)
                except ValueError as e:
                    print(f"Error reading line '{line}' in {file_path}: {e}")
                    continue

                data[binary_sequence] = data.get(binary_sequence, 0.0) + count
                total_count += count

            if show_progress:
                ProgressManager.update_progress(file_size)

    return data, total_count


def parse_parq(file_name, show_progress=False):
    """
    Reads a Parquet file and converts it back into a dictionary.

    Parameters:
        file_name (str): The Parquet file name to read.
        show_progress (bool, optional): Whether to display progress updates.

    Returns:
        dict: A dictionary where keys are states and values are probabilities.
    """
    total_steps = 2
    with (
        ProgressManager.progress("Parsing Parquet file", total_steps=total_steps)
        if show_progress
        else ProgressManager.dummy_context()
    ):
        # Step 1: Read the Parquet file into a DataFrame.
        df = pd.read_parquet(file_name, engine="pyarrow")
        if show_progress:
            ProgressManager.update_progress(1)

        # Step 2: Convert the DataFrame into a dictionary.
        data_dict = dict(zip(df["state"], df["probability"]))
        if show_progress:
            ProgressManager.update_progress(2)

    return data_dict


def parse_json(file_path, sorted=True, update_interval=10, show_progress=False):
    """
    Parse the Aquilla JSON file and return the binary sequence and its associated probability.
    Aquilla returns a JSON file which contains all the information on the submitted quantum task.
    This function only returns the states amd their probabilities.

    Args:
        file_path (str): Path to the input JSON file.
        sorted (bool, optional): Whether or not to remove presequences which have missing atoms.
        update_interval (int, optional): Number of lines before updating progress.
        show_progress (bool, optional): Whether to display progress updates.

    Returns:
        data (dict): A dictionary mapping binary sequences to their probabilities.
        total_count (float): The sum of number of shots.
    """
    data = {}
    total_count = 0.0

    with open(file_path, "r") as f:
        json_data = json.load(f)
    total_steps = len(json_data["measurements"])

    with (
        ProgressManager.progress("Parsing JSON file", total_steps)
        if show_progress
        else ProgressManager.dummy_context()
    ):

        for idx, line in enumerate(json_data["measurements"]):
            if show_progress and idx % update_interval == 0:
                ProgressManager.update_progress(idx + 1)
            try:
                pre_sequence = line["shotResult"]["preSequence"]
                # throw out the line if it is not a valid pre_sequence
                if sorted and sum(pre_sequence) != len(pre_sequence):
                    continue
                postSequence = line["shotResult"]["postSequence"]
                # invert binary (0 -> 1, 1 -> 0)
                postSequence = [1 - x for x in postSequence]
                bitString = "".join(str(x) for x in postSequence)
                total_count += 1
                if bitString in data:
                    data[bitString] += 1
                else:
                    data[bitString] = 1
            except Exception as e:
                print(f"Error reading measurement '{line}' in {file_path}: {e}")

    return data, total_count


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
