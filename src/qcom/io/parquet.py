"""
Parquet I/O
===========

Lightweight utilities to save and load probability distributions (bitstring -> probability)
using the Apache Parquet format via pandas.

Design notes
------------
- Uses `pyarrow` backend for Parquet (fast and well supported).
- Provides symmetry with JSON I/O (see `qcom.io.aquila`).
- Progress feedback integrated via ProgressManager.

Functions
---------
- parse_parquet(file_path, show_progress=False)
    → Load a parquet file into a dictionary {state: probability}.
- save_parquet(bitstring_probabilities, output_path)
    → Save a dictionary {state: probability} into a parquet file.
"""

from __future__ import annotations

import pandas as pd
from typing import Dict

from .._internal import ProgressManager
from .._internal.deprecations import warn_deprecated_alias

__all__ = ["parse_parquet", "save_parquet", "save_dict_to_parquet"]


def _install_pyarrow_hotfix() -> None:
    """
    Work around a pandas/pyarrow interaction where pandas tries to unregister
    an extension type that may not exist yet in newer pyarrow builds.

    The failure shows up as an ArrowKeyError during pandas.read_parquet() or
    DataFrame.to_parquet() before any file I/O happens.
    """
    try:
        import pyarrow
    except ImportError:
        return

    if getattr(pyarrow, "_qcom_hotfix_installed", False):
        return

    original_unregister = pyarrow.unregister_extension_type

    def safe_unregister_extension_type(name: str):
        try:
            return original_unregister(name)
        except pyarrow.ArrowKeyError:
            if name == "arrow.py_extension_type":
                return None
            raise

    pyarrow.unregister_extension_type = safe_unregister_extension_type
    pyarrow._qcom_hotfix_installed = True


def parse_parquet(
    file_path: str | None = None,
    show_progress: bool = False,
    *,
    file_name: str | None = None,
) -> Dict[str, float]:
    """
    Read a Parquet file into a {state: probability} dictionary.

    Parameters
    ----------
    file_path : str
        Path to the Parquet file to read.
    show_progress : bool, default False
        Whether to display progress updates.
    file_name : str | None, optional
        Compatibility keyword for older callers. Prefer `file_path`.

    Returns
    -------
    dict[str, float]
        Mapping from state strings to probabilities.
    """
    if file_path is None:
        if file_name is None:
            raise TypeError("parse_parquet: provide 'file_path'.")
        warn_deprecated_alias("parse_parquet(file_name=...)", "parse_parquet(file_path=...)")
        file_path = file_name
    elif file_name is not None:
        raise TypeError("parse_parquet: use either 'file_path' or 'file_name', not both.")

    total_steps = 2
    with (
        ProgressManager.progress("Parsing Parquet file", total_steps=total_steps)
        if show_progress
        else ProgressManager.dummy_context()
    ):
        _install_pyarrow_hotfix()
        df = pd.read_parquet(file_path, engine="pyarrow")
        if show_progress:
            ProgressManager.update_progress(1)

        bitstring_probabilities = dict(zip(df["state"], df["probability"]))
        if show_progress:
            ProgressManager.update_progress(2)

    return bitstring_probabilities


def save_parquet(bitstring_probabilities: Dict[str, float], output_path: str) -> None:
    """
    Save a dictionary {state: probability} to a Parquet file.

    Parameters
    ----------
    bitstring_probabilities : dict[str, float]
        Dictionary mapping states to probabilities.
    output_path : str
        Output Parquet filename.
    """
    total_steps = 3
    with ProgressManager.progress("Saving dictionary to Parquet", total_steps=total_steps):
        _install_pyarrow_hotfix()
        items = list(bitstring_probabilities.items())
        ProgressManager.update_progress(1)

        df = pd.DataFrame(items, columns=["state", "probability"])
        ProgressManager.update_progress(2)

        df.to_parquet(output_path, engine="pyarrow", index=False)
        ProgressManager.update_progress(3)

    print(f"Dictionary saved to {output_path}")


def save_dict_to_parquet(data_dict: Dict[str, float], file_name: str) -> None:
    """Deprecated compatibility alias for `save_parquet`."""
    warn_deprecated_alias("save_dict_to_parquet", "save_parquet")
    save_parquet(data_dict, output_path=file_name)
