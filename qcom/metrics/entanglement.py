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
- von_neumann_entropy_from_rdm(rdm):
    Compute the VNEE given a reduced density matrix.

- von_neumann_entropy_from_hamiltonian(hamiltonian, configuration, state_index=0, show_progress=False):
    Compute the VNEE of a subsystem from the ground or excited eigenstate
    of a Hamiltonian, given a partitioning specification.
"""

import numpy as np
from .._internal.progress import ProgressManager
from ..solvers.static import find_eigenstate


# -------------------- Von Neumann Entropy from RDM --------------------

def von_neumann_entropy_from_rdm(rdm: np.ndarray) -> float:
    """
    Compute the Von Neumann Entanglement Entropy (VNEE) from a reduced density matrix.

    Args:
        rdm (np.ndarray): Reduced density matrix of a subsystem.

    Returns:
        float: Von Neumann entropy, defined as -Tr(ρ log ρ).
    """
    eigenvalues = np.linalg.eigvalsh(rdm)
    eigenvalues = eigenvalues[eigenvalues > 0]  # filter out zero eigenvalues
    return -np.sum(eigenvalues * np.log(eigenvalues))


# -------------------- Von Neumann Entropy from Hamiltonian --------------------

def von_neumann_entropy_from_hamiltonian(
    hamiltonian,
    configuration,
    state_index: int = 0,
    show_progress: bool = False,
) -> float:
    """
    Compute VNEE from the eigenstate of a Hamiltonian with a given partition.

    Args:
        hamiltonian (np.ndarray or scipy.sparse matrix):
            Hamiltonian matrix. If sparse, it will be converted to dense.
        configuration (list[int]):
            Binary list specifying the bipartition of sites:
            - 1 for subsystem (kept)
            - 0 for environment (traced out)
        state_index (int, optional):
            Which eigenstate to use (0 = ground state, 1 = first excited, etc.).
            Default = 0.
        show_progress (bool, optional):
            Whether to show progress updates during computation.

    Returns:
        float: Von Neumann entanglement entropy of the specified subsystem.
    """
    if not isinstance(hamiltonian, np.ndarray):
        hamiltonian = hamiltonian.toarray()

    num_atoms = int(np.log2(hamiltonian.shape[0]))
    subsystem_atoms = [i for i, included in enumerate(configuration) if included == 1]
    subsystem_size = len(subsystem_atoms)

    total_steps = 5 + num_atoms  # decomposition + tracing + entropy
    step = 0

    with (
        ProgressManager.progress("Computing Von Neumann Entropy", total_steps)
        if show_progress
        else ProgressManager.dummy_context()
    ):
        # --- Eigen-decomposition (choose eigenstate) ---
        chosen_eigenvalue, chosen_state = find_eigenstate(
            hamiltonian, state_index, show_progress
        )
        step += 1
        if show_progress:
            ProgressManager.update_progress(min(step, total_steps))

        # --- Build full density matrix ρ = |ψ⟩⟨ψ| ---
        density_matrix = np.outer(chosen_state, chosen_state.conj())
        step += 1
        if show_progress:
            ProgressManager.update_progress(min(step, total_steps))

        # --- Reshape to tensor form for partial trace ---
        reshaped_matrix = density_matrix.reshape([2] * (2 * num_atoms))
        step += 1
        if show_progress:
            ProgressManager.update_progress(min(step, total_steps))

        # --- Partial trace out environment sites ---
        current_dim = num_atoms
        for atom in reversed(range(num_atoms)):
            if configuration[atom] == 0:  # trace out this site
                reshaped_matrix = np.trace(
                    reshaped_matrix, axis1=atom, axis2=atom + current_dim
                )
                current_dim -= 1
                step += 1
                if show_progress:
                    ProgressManager.update_progress(min(step, total_steps))

        # --- Reshape back to reduced density matrix ---
        dim_subsystem = 2**subsystem_size
        reduced_density_matrix = reshaped_matrix.reshape((dim_subsystem, dim_subsystem))
        step += 1
        if show_progress:
            ProgressManager.update_progress(min(step, total_steps))

        # --- Compute entropy ---
        entropy = von_neumann_entropy_from_rdm(reduced_density_matrix)
        if show_progress:
            ProgressManager.update_progress(total_steps)

    return entropy