"""
Aquila JSON I/O
===============

Lightweight loader for **QuEra Aquila** JSON task results. This reader extracts
the computational-basis outcomes from the `measurements[*].shotResult` records
and returns (counts, total_shots).

Design Notes
------------
- Minimal dependencies (stdlib only); optional progress via ProgressManager.
- Supports filtering out shots with incomplete `preSequence`.
- Inverts `postSequence` bits (0↔1) to match the historical QCOM convention.

Returns
-------
- data : dict[str, float]
    Mapping bitstring → count (float for historical compatibility with old code).
- total_count : float
    Total number of accepted shots.

Deprecated compatibility aliases keep the legacy `parse_json(..., sorted=...)` entrypoint.
"""

from __future__ import annotations

import json
from typing import Tuple, Dict

from .._internal import ProgressManager
from .._internal.deprecations import warn_deprecated_alias

__all__ = ["parse_aquila_json", "parse_json"]


def parse_aquila_json(
    file_path: str,
    filter_incomplete: bool = True,
    update_interval: int = 10,
    show_progress: bool = False,
) -> Tuple[Dict[str, float], float]:
    """
    Parse an Aquila JSON file and return (counts_dict, total_shots).

    Parameters
    ----------
    file_path : str
        Path to the Aquila JSON file.
    filter_incomplete : bool, default True
        If True, discard measurements whose `preSequence` contains missing atoms.
    update_interval : int, default 10
        Frequency (in records) for progress updates when `show_progress=True`.
    show_progress : bool, default False
        If True, display a progress bar using ProgressManager.

    Returns
    -------
    data : dict[str, float]
        Mapping from bitstring to raw counts.
    total_count : float
        Total number of accepted shots.
    """
    counts: Dict[str, float] = {}
    total_count: float = 0.0

    # Load once (entire file typically fits in memory for Aquila result sizes)
    with open(file_path, "r") as f:
        json_data = json.load(f)

    measurements = json_data.get("measurements", [])
    total_steps = len(measurements)

    with (
        ProgressManager.progress("Parsing JSON file", total_steps)
        if show_progress
        else ProgressManager.dummy_context()
    ):
        for idx, record in enumerate(measurements):
            if show_progress and (idx % update_interval == 0):
                ProgressManager.update_progress(idx + 1)

            try:
                shot = record["shotResult"]
                pre_sequence = shot["preSequence"]

                # Optionally discard shots with any missing/invalid atoms
                if filter_incomplete and sum(pre_sequence) != len(pre_sequence):
                    continue

                post_sequence = shot["postSequence"]

                # Historical convention: invert bits (0→1, 1→0)
                post_sequence = [1 - x for x in post_sequence]
                bit_string = "".join(str(x) for x in post_sequence)

                total_count += 1.0
                counts[bit_string] = counts.get(bit_string, 0.0) + 1.0

            except Exception as e:
                # Keep parsing other lines; surface the offending record for debugging
                print(f"Error reading measurement #{idx} in {file_path}: {e}")

        if show_progress:
            ProgressManager.update_progress(total_steps)

    return counts, total_count


def parse_json(
    file_path: str,
    sorted: bool = True,  # keep legacy name for compatibility
    filter_incomplete: bool | None = None,
    update_interval: int = 10,
    show_progress: bool = False,
) -> Tuple[Dict[str, float], float]:
    """Deprecated compatibility alias for `parse_aquila_json`."""
    warn_deprecated_alias("parse_json", "parse_aquila_json")
    return parse_aquila_json(
        file_path,
        filter_incomplete=sorted if filter_incomplete is None else bool(filter_incomplete),
        update_interval=update_interval,
        show_progress=show_progress,
    )
