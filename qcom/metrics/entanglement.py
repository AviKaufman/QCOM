# qcom/metrics/entanglement.py
"""
qcom.metrics.entanglement
=========================

Quantum entanglement measures on simulated states.

Purpose
-------
Provide utilities to compute the Von Neumann entanglement entropy (VNEE)
for reduced subsystems, either from a reduced density matrix (RDM) directly
or derived from a Hamiltonian and eigenstate.

Functions
---------
- von_neumann_entropy_from_rdm(rdm, base=np.e):
    Compute the VNEE given a reduced density matrix.

- von_neumann_entropy_from_hamiltonian(hamiltonian, configuration, state_index=0, show_progress=False, base=np.e):
    Compute the VNEE of a subsystem from the ground or excited eigenstate
    of a Hamiltonian, given a partitioning specification.

- von_neumann_entropy_from_state(psi, configuration, show_progress=False, base=np.e):
    Compute the VNEE of a subsystem directly from a state vector.
"""

from __future__ import annotations

import numpy as np
from .._internal.progress import ProgressManager
from ..solvers.static import find_eigenstate, as_linear_operator
from .states import create_density_matrix, compute_reduced_density_matrix


# -------------------- Von Neumann Entropy from RDM --------------------

def von_neumann_entropy_from_rdm(rdm: np.ndarray, *, base: float = 2) -> float:
    """
    Compute the Von Neumann Entanglement Entropy (VNEE) from a reduced density matrix.

    Args:
        rdm: Reduced density matrix of a subsystem (square Hermitian, trace ~ 1).
        base: Logarithm base. Use `np.e` for nats, or `2` for bits (default).

    Returns:
        float: Von Neumann entropy, defined as -Tr(ρ log ρ) / log(base).
    """
    # Hermitian eigvals; filter out non-positive values to avoid log(0/neg)
    evals = np.linalg.eigvalsh(rdm)
    evals = evals[evals > 0.0]
    if evals.size == 0:
        return 0.0
    # Compute in natural logs and convert base at the end for numerical stability
    entropy_nats = -np.sum(evals * np.log(evals))
    if base == np.e:
        return float(entropy_nats)
    return float(entropy_nats / np.log(base))


# -------------------- Von Neumann Entropy from Hamiltonian --------------------

def von_neumann_entropy_from_hamiltonian(
    hamiltonian,
    configuration: list[int] | np.ndarray,
    *,
    state_index: int = 0,
    show_progress: bool = False,
    base: float = 2,
) -> float:
    """
    Compute VNEE from the eigenstate of a Hamiltonian with a given partition.

    This implementation **does not densify the Hamiltonian**: it uses the
    static solver to obtain the eigenstate directly from the provided backend
    (dense array, SciPy sparse, or LinearOperator), then forms the reduced
    density matrix via a tensor reshape + partial trace.

    Args:
        hamiltonian:
            Hamiltonian-like object supported by `find_eigenstate` and
            `as_linear_operator` (dense ndarray, SciPy sparse, or LinearOperator).
        configuration:
            Binary list/array specifying the bipartition of sites:
            - 1 → keep (subsystem A)
            - 0 → trace out (environment)
        state_index:
            Which eigenstate to use (0 = ground state, 1 = first excited, ...).
        show_progress:
            Whether to show progress updates during computation.
        base:
            Logarithm base for entropy (np.e for nats, 2 for bits).

    Returns:
        float: Von Neumann entanglement entropy of the specified subsystem.
    """
    # Determine Hilbert dimension robustly without materializing a dense matrix
    _, _, dim = as_linear_operator(hamiltonian)
    if dim <= 0:
        raise ValueError("Could not infer a positive Hilbert dimension from 'hamiltonian'.")

    # Check power-of-two dimension and deduce qubit count
    n_qubits = int(np.round(np.log2(dim)))
    if (1 << n_qubits) != dim:
        raise ValueError(f"Hilbert space dimension {dim} is not a power of 2.")
    if len(configuration) != n_qubits:
        raise ValueError("Length of 'configuration' must equal the number of qubits.")

    # Progress bookkeeping: eigenstate + build rho + reshape + trace loop + final reshape + entropy
    total_steps = 4 + n_qubits
    step = 0

    with (
        ProgressManager.progress("Computing Von Neumann Entropy", total_steps)
        if show_progress
        else ProgressManager.dummy_context()
    ):
        # --- 1) Eigen-decomposition (choose eigenstate) ---
        _, psi = find_eigenstate(hamiltonian, state_index=state_index, show_progress=show_progress)
        step += 1
        if show_progress:
            ProgressManager.update_progress(min(step, total_steps))

        # --- 2) Full density matrix ρ = |ψ⟩⟨ψ| ---
        rho = create_density_matrix(np.asarray(psi).reshape(-1), show_progress=False)
        step += 1
        if show_progress:
            ProgressManager.update_progress(min(step, total_steps))

        # --- 3) Partial trace to reduced density matrix on selected subsystem ---
        # configuration convention: 1 = keep, 0 = trace out
        rdm = compute_reduced_density_matrix(rho, configuration, show_progress=False)
        step += n_qubits  # account for worst-case tracing loop inside helper
        if show_progress:
            ProgressManager.update_progress(min(step, total_steps))

        # --- 4) Entropy on RDM ---
        S = von_neumann_entropy_from_rdm(rdm, base=base)
        if show_progress:
            ProgressManager.update_progress(total_steps)

    return S


# -------------------- Von Neumann Entropy directly from state --------------------

def von_neumann_entropy_from_state(
    psi: np.ndarray,
    configuration: list[int] | np.ndarray,
    *,
    show_progress: bool = False,
    base: float = 2,
) -> float:
    """
    Compute VNEE of a subsystem directly from a state vector |ψ⟩.

    Args:
        psi:
            State vector of length 2^N (complex). Will be reshaped to 1D.
        configuration:
            Binary list/array specifying the bipartition of sites:
            - 1 → keep (subsystem A)
            - 0 → trace out (environment)
        show_progress:
            Whether to show progress updates during computation.
        base:
            Logarithm base for entropy (np.e for nats, 2 for bits).

    Returns:
        float: Von Neumann entanglement entropy of the specified subsystem.
    """
    psi = np.asarray(psi).reshape(-1)
    dim = psi.shape[0]
    if dim == 0 or (dim & (dim - 1)) != 0:
        raise ValueError(f"State vector length {dim} is not a power of 2.")
    n_qubits = int(np.log2(dim))
    if len(configuration) != n_qubits:
        raise ValueError("Length of 'configuration' must equal the number of qubits.")

    total_steps = 3 + n_qubits
    step = 0

    with (
        ProgressManager.progress("Computing Von Neumann Entropy (from state)", total_steps)
        if show_progress
        else ProgressManager.dummy_context()
    ):
        # --- 1) ρ = |ψ⟩⟨ψ| ---
        rho = create_density_matrix(psi, show_progress=False)
        step += 1
        if show_progress:
            ProgressManager.update_progress(min(step, total_steps))

        # --- 2) Partial trace to RDM ---
        rdm = compute_reduced_density_matrix(rho, configuration, show_progress=False)
        step += n_qubits
        if show_progress:
            ProgressManager.update_progress(min(step, total_steps))

        # --- 3) Entropy on RDM ---
        S = von_neumann_entropy_from_rdm(rdm, base=base)
        if show_progress:
            ProgressManager.update_progress(total_steps)

    return S