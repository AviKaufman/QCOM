"""Sampling and dataset-combination utilities for QCOM."""

import random
from collections.abc import Mapping
from typing import Union

from qcom.core import CountsData, ProbabilityData

from .._internal.deprecations import warn_deprecated_alias
from .._internal import ProgressManager
from ..data.ops import normalize_to_probabilities

__all__ = [
    "sample_counts",
    "combine_bitstring_datasets",
    "sample_data",
    "combine_datasets",
]


def sample_counts(
    counts: dict[str, int] | CountsData,
    total_count: int,
    sample_size: int,
    update_interval: int = 100,
    show_progress: bool = False,
) -> dict[str, int]:
    """
    Sample bitstrings from raw counts according to their probabilities.

    Args:
        counts: Bitstring counts.
        total_count: Total count used for normalization.
        sample_size: Number of samples to generate.
        update_interval: Frequency of progress updates when progress is enabled.
        show_progress: Whether to display progress updates.

    Returns:
        New bitstring counts drawn from the normalized distribution.
    """
    normalized_probabilities = normalize_to_probabilities(counts, total_count)
    if isinstance(normalized_probabilities, ProbabilityData):
        normalized_probabilities = normalized_probabilities.to_dict()
    bitstrings = list(normalized_probabilities.keys())
    probabilities = list(normalized_probabilities.values())

    sampled_counts: dict[str, int] = {}

    with (
        ProgressManager.progress("Sampling counts", total_steps=sample_size)
        if show_progress
        else ProgressManager.dummy_context()
    ):
        sampled_bitstrings = random.choices(bitstrings, weights=probabilities, k=sample_size)

        for index, bitstring in enumerate(sampled_bitstrings):
            sampled_counts[bitstring] = sampled_counts.get(bitstring, 0) + 1
            if show_progress and index % update_interval == 0:
                ProgressManager.update_progress(index + 1)

        if show_progress:
            ProgressManager.update_progress(sample_size)

    return sampled_counts


def combine_bitstring_datasets(
    left_dataset: Mapping[str, Union[int, float]] | CountsData | ProbabilityData,
    right_dataset: Mapping[str, Union[int, float]] | CountsData | ProbabilityData,
    tol: float = 1e-6,
    update_interval: int = 100,
    show_progress: bool = False,
    return_data: bool = False,
) -> dict[str, Union[int, float]] | CountsData | ProbabilityData:
    """
    Combine two compatible bitstring datasets.

    Rules:
        - Two probability datasets are merged and renormalized.
        - Two count datasets are merged directly.
        - Mixed count/probability inputs raise an error.
    """
    source: str | None = None
    left_values: Mapping[str, Union[int, float]]
    right_values: Mapping[str, Union[int, float]]
    if isinstance(left_dataset, CountsData) or isinstance(right_dataset, CountsData):
        if not isinstance(left_dataset, CountsData) or not isinstance(right_dataset, CountsData):
            raise ValueError(
                "Cannot combine CountsData with probabilities or raw dictionaries. "
                "Convert both inputs to the same explicit container type first."
            )
        dataset_kind = "counts"
        left_values = left_dataset.to_dict()
        right_values = right_dataset.to_dict()
        source = left_dataset.source or right_dataset.source
    elif isinstance(left_dataset, ProbabilityData) or isinstance(right_dataset, ProbabilityData):
        if not isinstance(left_dataset, ProbabilityData) or not isinstance(
            right_dataset, ProbabilityData
        ):
            raise ValueError(
                "Cannot combine ProbabilityData with counts or raw dictionaries. "
                "Convert both inputs to the same explicit container type first."
            )
        dataset_kind = "probabilities"
        left_values = left_dataset.to_dict()
        right_values = right_dataset.to_dict()
        source = left_dataset.source or right_dataset.source
    else:
        left_values = dict(left_dataset)
        right_values = dict(right_dataset)

        left_total = sum(left_values.values())
        right_total = sum(right_values.values())

        left_is_probability = abs(left_total - 1.0) < tol
        right_is_probability = abs(right_total - 1.0) < tol

        if left_is_probability and right_is_probability:
            dataset_kind = "probabilities"
        elif (left_is_probability and not right_is_probability) or (
            not left_is_probability and right_is_probability
        ):
            raise ValueError(
                "Cannot combine a dataset of probabilities with a dataset of counts. "
                "Convert one to the other before combining."
            )
        else:
            dataset_kind = "counts"

    combined_values: dict[str, Union[int, float]] = {}
    all_bitstrings = set(left_values.keys()).union(right_values.keys())
    total_bitstrings = len(all_bitstrings)

    with (
        ProgressManager.progress("Combining bitstring datasets", total_steps=total_bitstrings)
        if show_progress
        else ProgressManager.dummy_context()
    ):
        for index, bitstring in enumerate(all_bitstrings):
            combined_values[bitstring] = left_values.get(bitstring, 0) + right_values.get(
                bitstring, 0
            )

            if show_progress and index % update_interval == 0:
                ProgressManager.update_progress(index + 1)

        if show_progress:
            ProgressManager.update_progress(total_bitstrings)

    if dataset_kind == "probabilities":
        combined_total = sum(combined_values.values())
        combined_values = {
            bitstring: value / combined_total for bitstring, value in combined_values.items()
        }
        if return_data:
            return ProbabilityData(combined_values, source=source)
    elif return_data:
        return CountsData(
            {bitstring: int(value) for bitstring, value in combined_values.items()},
            source=source,
        )

    return combined_values


def sample_data(
    data: dict[str, int] | CountsData,
    total_count: int,
    sample_size: int,
    update_interval: int = 100,
    show_progress: bool = False,
) -> dict[str, int]:
    """Deprecated compatibility alias for `sample_counts`."""
    warn_deprecated_alias("sample_data", "sample_counts")
    return sample_counts(
        data,
        total_count=total_count,
        sample_size=sample_size,
        update_interval=update_interval,
        show_progress=show_progress,
    )


def combine_datasets(
    data1: Mapping[str, Union[int, float]] | CountsData | ProbabilityData,
    data2: Mapping[str, Union[int, float]] | CountsData | ProbabilityData,
    tol: float = 1e-6,
    update_interval: int = 100,
    show_progress: bool = False,
    return_data: bool = False,
) -> dict[str, Union[int, float]] | CountsData | ProbabilityData:
    """Deprecated compatibility alias for `combine_bitstring_datasets`."""
    warn_deprecated_alias("combine_datasets", "combine_bitstring_datasets")
    return combine_bitstring_datasets(
        data1,
        data2,
        tol=tol,
        update_interval=update_interval,
        show_progress=show_progress,
        return_data=return_data,
    )
