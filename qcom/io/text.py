# qcom/io/text.py
"""
Plaintext I/O
=============

Utilities for parsing and saving simple text-based probability/count files.
Format is line-based with whitespace separation:

    <state> <value>

where <state> is a bitstring (e.g., "0101") and <value> is a float count/probability.

Design notes
------------
- Compatible with QCOM's JSON and Parquet loaders/savers.
- Supports progress reporting via ProgressManager.
- Intended for lightweight debugging or inspection (human-readable).

Functions
---------
- parse_file(file_path, update_interval=500000, show_progress=False)
    → Load a text file into a dictionary {state: count}, with total count returned separately.
- save_data(data, savefile, update_interval=100, show_progress=False)
    → Save a dictionary {state: count/prob} back to text format.
"""

from __future__ import annotations

# -------------------- Imports --------------------
from .._internal import ProgressManager
import os
from typing import Dict, Tuple


# -------------------- Text reader --------------------
def parse_file(
    file_path: str,
    update_interval: int = 500_000,
    show_progress: bool = False,
) -> Tuple[Dict[str, float], float]:
    """
    Parse a whitespace-delimited text file into counts.

    Parameters
    ----------
    file_path : str
        Path to the input file.
    update_interval : int, default 500000
        Frequency (in lines) at which to update progress if enabled.
    show_progress : bool, default False
        Whether to display progress updates.

    Returns
    -------
    data : dict[str, float]
        Mapping from bitstring states to raw counts.
    total_count : float
        Sum of all counts across the file.
    """
    data: Dict[str, float] = {}
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


# -------------------- Text writer --------------------
def save_data(
    data: Dict[str, float],
    savefile: str,
    update_interval: int = 100,
    show_progress: bool = False,
) -> None:
    """
    Save data to a whitespace-delimited text file.

    Format:
        <state> <value>
    where 'state' is the bitstring and 'value' is the count or probability.

    Parameters
    ----------
    data : dict[str, float]
        Mapping from states to counts/probabilities.
    savefile : str
        Path to the file where the data will be written.
    update_interval : int, default 100
        Frequency (in states) at which to update progress if enabled.
    show_progress : bool, default False
        Whether to display progress updates.
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