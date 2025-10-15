# qcom/io/parquet.py
"""
Parquet I/O
===========

Lightweight utilities to save and load probability distributions (bitstring → prob)
using the Apache Parquet format via pandas.

Design notes
------------
- Uses `pyarrow` backend for Parquet (fast and well supported).
- Provides symmetry with JSON I/O (see `qcom.io.aquila`).
- Progress feedback integrated via ProgressManager.

Functions
---------
- parse_parq(file_name, show_progress=False)
    → Load a parquet file into a dictionary {state: probability}.
- save_dict_to_parquet(data_dict, file_name)
    → Save a dictionary {state: probability} into a parquet file.
"""

from __future__ import annotations

# -------------------- Imports --------------------
from .._internal import ProgressManager
import pandas as pd
from typing import Dict


# -------------------- Parquet reader --------------------
def parse_parquet(file_name: str, show_progress: bool = False) -> Dict[str, float]:
    """
    Read a Parquet file into a {state: probability} dictionary.

    Parameters
    ----------
    file_name : str
        Path to the Parquet file to read.
    show_progress : bool, default False
        Whether to display progress updates.

    Returns
    -------
    dict[str, float]
        Mapping from state strings to probabilities.
    """
    total_steps = 2
    with (
        ProgressManager.progress("Parsing Parquet file", total_steps=total_steps)
        if show_progress
        else ProgressManager.dummy_context()
    ):
        # --- Step 1: Read file into DataFrame ---
        df = pd.read_parquet(file_name, engine="pyarrow")
        if show_progress:
            ProgressManager.update_progress(1)

        # --- Step 2: Convert DataFrame to dictionary ---
        data_dict = dict(zip(df["state"], df["probability"]))
        if show_progress:
            ProgressManager.update_progress(2)

    return data_dict


# -------------------- Parquet writer --------------------
def save_dict_to_parquet(data_dict: Dict[str, float], file_name: str) -> None:
    """
    Save a dictionary {state: probability} to a Parquet file.

    Parameters
    ----------
    data_dict : dict[str, float]
        Dictionary mapping states to probabilities.
    file_name : str
        Output Parquet filename.
    """
    total_steps = 3
    with ProgressManager.progress(
        "Saving dictionary to Parquet", total_steps=total_steps
    ):
        # --- Step 1: Convert dict to list of items ---
        items = list(data_dict.items())
        ProgressManager.update_progress(1)

        # --- Step 2: Create DataFrame ---
        df = pd.DataFrame(items, columns=["state", "probability"])
        ProgressManager.update_progress(2)

        # --- Step 3: Write to Parquet file ---
        df.to_parquet(file_name, engine="pyarrow", index=False)
        ProgressManager.update_progress(3)

    print(f"Dictionary saved to {file_name}")