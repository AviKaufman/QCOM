# qcom/metrics/__init__.py
"""
qcom.metrics
============

Collection of analysis routines for classical and quantum information metrics.

Purpose
-------
This subpackage provides reusable functions to compute and manipulate
probabilities, entropies, density matrices, and related measures. It
complements the Hamiltonian builders by exposing tools for analyzing their
outputs.

Modules
-------
- bitstrings   : Helpers for manipulating binary-string dictionaries
                 (ordering, partitioning, etc.).
- classical    : Shannon entropy, mutual information, conditional entropy.
- entanglement : Von Neumann entanglement entropy from Hamiltonians or RDMs.
- probabilities: Cumulative distributions, N(p) metrics, eigenstate probabilities.
- states       : Density matrix construction and reduced density matrix (partial trace).

Typical usage
-------------
>>> from qcom.metrics import compute_shannon_entropy, von_neumann_entropy_from_hamiltonian
>>> H = some_hamiltonian.to_dense()
>>> vnee = von_neumann_entropy_from_hamiltonian(H, configuration=[1,0,1,0])
>>> shannon = compute_shannon_entropy(prob_dict)
"""

from __future__ import annotations

# Re-export selected top-level functions for convenience
from .bitstrings import order_dict, part_dict
from .classical import (
    compute_shannon_entropy,
    compute_reduced_shannon_entropy,
    compute_mutual_information,
    compute_conditional_entropy,
)
from .entanglement import (
    von_neumann_entropy_from_rdm,
    von_neumann_entropy_from_hamiltonian,
)
from .probabilities import (
    cumulative_probability_at_value,
    cumulative_distribution,
    compute_N_of_p_all,
    compute_N_of_p,
    get_eigenstate_probabilities,
)
from .states import (
    create_density_matrix,
    compute_reduced_density_matrix,
)

__all__ = [
    # bitstrings
    "order_dict",
    "part_dict",
    # classical
    "compute_shannon_entropy",
    "compute_reduced_shannon_entropy",
    "compute_mutual_information",
    "compute_conditional_entropy",
    # entanglement
    "von_neumann_entropy_from_rdm",
    "von_neumann_entropy_from_hamiltonian",
    # probabilities
    "cumulative_probability_at_value",
    "cumulative_distribution",
    "compute_N_of_p_all",
    "compute_N_of_p",
    "get_eigenstate_probabilities",
    # states
    "create_density_matrix",
    "compute_reduced_density_matrix",
]