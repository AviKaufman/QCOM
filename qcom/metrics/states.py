# qcom/metrics/states.py
"""
qcom.metrics.states
===================

Basic quantum-state utilities.

Purpose
-------
Provide small, NumPy-first helpers for:
- Building a pure-state density matrix ρ = |ψ⟩⟨ψ|.
- Computing a reduced density matrix via partial trace over a computational basis
  bipartition specified by a binary configuration vector.

Notes
-----
- These routines assume qubit systems and computational-basis ordering.
- The partial trace implementation follows the standard reshape+trace pattern.
"""

from __future__ import annotations

import time
import numpy as np
from .._internal.progress import ProgressManager


# -------------------- Construct ρ from a state vector --------------------

def create_density_matrix(eigenvector: np.ndarray, show_progress: bool = False) -> np.ndarray:
    """
    Construct the pure-state density matrix ρ = |ψ⟩⟨ψ|.

    Args:
        eigenvector: 1D complex/real NumPy array |ψ⟩ of length 2^N.
        show_progress: If True, display a one-step progress message.

    Returns:
        2D NumPy array ρ with shape (2^N, 2^N).
    """
    with (
        ProgressManager.progress("Constructing Density Matrix", 1)
        if show_progress
        else ProgressManager.dummy_context()
    ):
        rho = np.outer(eigenvector, np.conj(eigenvector))
        if show_progress:
            ProgressManager.update_progress(1)
    return rho


# -------------------- Reduced density matrix via partial trace --------------------

def compute_reduced_density_matrix(
    density_matrix: np.ndarray,
    configuration: list[int] | np.ndarray,
    show_progress: bool = False,
) -> np.ndarray:
    """
    Compute the reduced density matrix by tracing out sites marked 0 in `configuration`.

    Convention
    ----------
    `configuration` is a length-N binary list/array:
        - 1 → keep this site (subsystem)
        - 0 → trace this site out (environment)

    Args:
        density_matrix: Full density matrix with shape (2^N, 2^N).
        configuration: Binary vector of length N selecting the kept sites.
        show_progress: If True, display progress updates.

    Returns:
        Reduced density matrix ρ_A with shape (2^|A|, 2^|A|),
        where A is the set of indices with configuration[i] == 1.
    """
    num_qubits = int(np.log2(density_matrix.shape[0]))
    if density_matrix.shape != (1 << num_qubits, 1 << num_qubits):
        raise ValueError("density_matrix must be square with dimension 2^N.")
    if len(configuration) != num_qubits:
        raise ValueError("configuration length must equal number of qubits N.")

    subsystem_atoms = [i for i, included in enumerate(configuration) if included == 1]
    subsystem_size = len(subsystem_atoms)

    total_steps = 2 + num_qubits
    step = 0

    with (
        ProgressManager.progress("Computing Reduced Density Matrix", total_steps)
        if show_progress
        else ProgressManager.dummy_context()
    ):
        if show_progress:
            print(
                "\rReshaping Density Matrix for Partial Trace... Please wait.",
                end="",
                flush=True,
            )
        start_time = time.time()

        # Reshape ρ into a rank-2N tensor with indices (i1..iN, j1..jN)
        reshaped = density_matrix.reshape([2] * (2 * num_qubits))
        step += 1
        if show_progress:
            ProgressManager.update_progress(min(step, total_steps))

        # Trace out qubits with configuration == 0 by contracting i_k with j_k
        current_dim = num_qubits
        for atom in reversed(range(num_qubits)):
            if configuration[atom] == 0:
                reshaped = np.trace(reshaped, axis1=atom, axis2=atom + current_dim)
                current_dim -= 1
                step += 1
                if show_progress:
                    ProgressManager.update_progress(min(step, total_steps))

        # Flatten back to a matrix on the kept subsystem
        dim_subsystem = 1 << subsystem_size
        reduced_density_matrix = reshaped.reshape((dim_subsystem, dim_subsystem))
        step += 1
        if show_progress:
            ProgressManager.update_progress(min(step, total_steps))

        if show_progress:
            end_time = time.time()
            print("\r" + " " * 80, end="")
            print(
                f"\rReduced Density Matrix computed in {end_time - start_time:.2f} seconds.",
                flush=True,
            )
            ProgressManager.update_progress(total_steps)

    return reduced_density_matrix