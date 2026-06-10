"""
QCOM public facade.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__version__ = "0.2.2"
__author__ = "Avi Kaufman"

__all__ = [
    "__version__",
    "controls",
    "core",
    "data",
    "hamiltonians",
    "io",
    "metrics",
    "solvers",
    "viz",
    "CountsData",
    "EvolutionResult",
    "LatticeRegister",
    "MutualInformationResult",
    "ProbabilityData",
    "RydbergAdapter",
    "SpectrumResult",
    "TimeSeries",
    "build_ising",
    "build_rydberg",
    "combine_datasets",
    "compute_conditional_entropy",
    "compute_mutual_information",
    "compute_reduced_shannon_entropy",
    "compute_shannon_entropy",
    "cumulative_distribution",
    "find_eigenstate",
    "get_eigenstate_probabilities",
    "ground_state",
    "introduce_error",
    "m3_mitigate_counts_from_rates",
    "normalize_to_probabilities",
    "parse_file",
    "parse_json",
    "parse_parquet",
    "print_most_probable_data",
    "sample_data",
    "save_data",
    "statevector_to_probabilities",
    "truncate_probabilities",
    "von_neumann_entropy_from_state",
]

_SUBMODULES = {"controls", "core", "data", "hamiltonians", "io", "metrics", "solvers", "viz"}

_EXPORTS: dict[str, tuple[str, str]] = {
    "CountsData": ("qcom.core", "CountsData"),
    "EvolutionResult": ("qcom.core", "EvolutionResult"),
    "MutualInformationResult": ("qcom.core", "MutualInformationResult"),
    "ProbabilityData": ("qcom.core", "ProbabilityData"),
    "SpectrumResult": ("qcom.core", "SpectrumResult"),
    "TimeSeries": ("qcom.controls.time_series", "TimeSeries"),
    "RydbergAdapter": ("qcom.controls.adapters.rydberg", "RydbergAdapter"),
    "introduce_error": ("qcom.data.noise", "introduce_error"),
    "m3_mitigate_counts_from_rates": ("qcom.data.noise", "m3_mitigate_counts_from_rates"),
    "normalize_to_probabilities": ("qcom.data.ops", "normalize_to_probabilities"),
    "truncate_probabilities": ("qcom.data.ops", "truncate_probabilities"),
    "print_most_probable_data": ("qcom.data.ops", "print_most_probable_data"),
    "sample_data": ("qcom.data.sampling", "sample_data"),
    "combine_datasets": ("qcom.data.sampling", "combine_datasets"),
    "build_ising": ("qcom.hamiltonians", "build_ising"),
    "build_rydberg": ("qcom.hamiltonians", "build_rydberg"),
    "parse_json": ("qcom.io", "parse_json"),
    "parse_parquet": ("qcom.io", "parse_parquet"),
    "parse_file": ("qcom.io", "parse_file"),
    "save_data": ("qcom.io", "save_data"),
    "compute_shannon_entropy": ("qcom.metrics", "compute_shannon_entropy"),
    "compute_mutual_information": ("qcom.metrics", "compute_mutual_information"),
    "compute_conditional_entropy": ("qcom.metrics", "compute_conditional_entropy"),
    "compute_reduced_shannon_entropy": ("qcom.metrics", "compute_reduced_shannon_entropy"),
    "von_neumann_entropy_from_state": (
        "qcom.metrics.entanglement",
        "von_neumann_entropy_from_state",
    ),
    "cumulative_distribution": ("qcom.metrics", "cumulative_distribution"),
    "get_eigenstate_probabilities": ("qcom.metrics", "get_eigenstate_probabilities"),
    "statevector_to_probabilities": ("qcom.metrics.probabilities", "statevector_to_probabilities"),
    "ground_state": ("qcom.solvers.static", "ground_state"),
    "find_eigenstate": ("qcom.solvers.static", "find_eigenstate"),
    "LatticeRegister": ("qcom.lattice_register", "LatticeRegister"),
}

if TYPE_CHECKING:
    from .controls.time_series import TimeSeries
    from .core import (
        CountsData,
        EvolutionResult,
        MutualInformationResult,
        ProbabilityData,
        SpectrumResult,
    )
    from .lattice_register import LatticeRegister


def __getattr__(name: str):
    if name in _SUBMODULES:
        import importlib

        module = importlib.import_module(f"qcom.{name}")
        globals()[name] = module
        return module
    if name in _EXPORTS:
        import importlib

        module_name, attr = _EXPORTS[name]
        value = getattr(importlib.import_module(module_name), attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'qcom' has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
