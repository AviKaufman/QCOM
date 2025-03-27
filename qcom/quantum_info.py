import numpy as np
from .progress import ProgressManager
from utils import find_eigenstate
import time


"""
Functions for computing quantum information measures in quantum systems. This will be an every growing list of functions.
"""


def von_neumann_entropy_from_rdm(rdm):
    """Computes the Von Neumann Entanglement Entropy given a reduced density matrix."""
    eigenvalues = np.linalg.eigvalsh(rdm)
    eigenvalues = eigenvalues[eigenvalues > 0]  # Avoid log(0)
    return -np.sum(eigenvalues * np.log(eigenvalues))


def von_neumann_entropy_from_hamiltonian(
    hamiltonian, configuration, state_index=0, show_progress=False
):
    """Computes VNEE given a Hamiltonian and partition specification."""
    if not isinstance(hamiltonian, np.ndarray):
        hamiltonian = hamiltonian.toarray()  # Convert sparse to dense

    num_atoms = int(np.log2(hamiltonian.shape[0]))  # Number of atoms in the system
    subsystem_atoms = [i for i, included in enumerate(configuration) if included == 1]
    subsystem_size = len(subsystem_atoms)
    total_steps = (
        5 + num_atoms
    )  # Decomposition, reshaping, tracing steps, and entropy computation
    step = 0

    with (
        ProgressManager.progress("Computing Von Neumann Entropy", total_steps)
        if show_progress
        else ProgressManager.dummy_context()
    ):
        chosen_eigenvalue, chosen_state = find_eigenstate(
            hamiltonian, state_index, show_progress
        )
        step += 1
        if show_progress:
            ProgressManager.update_progress(min(step, total_steps))

        density_matrix = np.outer(chosen_state, chosen_state.conj())
        step += 1
        if show_progress:
            ProgressManager.update_progress(min(step, total_steps))

        reshaped_matrix = density_matrix.reshape([2] * (2 * num_atoms))
        step += 1
        if show_progress:
            ProgressManager.update_progress(min(step, total_steps))

        current_dim = num_atoms
        for atom in reversed(range(num_atoms)):
            if configuration[atom] == 0:
                reshaped_matrix = np.trace(
                    reshaped_matrix, axis1=atom, axis2=atom + current_dim
                )
                current_dim -= 1
                step += 1
                if show_progress:
                    ProgressManager.update_progress(min(step, total_steps))

        dim_subsystem = 2**subsystem_size
        reduced_density_matrix = reshaped_matrix.reshape((dim_subsystem, dim_subsystem))
        step += 1
        if show_progress:
            ProgressManager.update_progress(min(step, total_steps))

        entropy = von_neumann_entropy_from_rdm(reduced_density_matrix)
        if show_progress:
            ProgressManager.update_progress(total_steps)

    return entropy


def get_eigenstate_probabilities(hamiltonian, state_index=0, show_progress=False):
    """
    Computes the probability distribution of the chosen eigenstate in the computational basis.
    """
    if not isinstance(hamiltonian, np.ndarray):
        hamiltonian = hamiltonian.toarray()

    num_qubits = int(np.log2(hamiltonian.shape[0]))
    hilbert_dim = 2**num_qubits
    total_steps = 4 + hilbert_dim
    step = 0

    with (
        ProgressManager.progress("Computing Eigenstate Probabilities", total_steps)
        if show_progress
        else ProgressManager.dummy_context()
    ):
        chosen_eigenvalue, chosen_state = find_eigenstate(
            hamiltonian, state_index, show_progress
        )
        step += 1
        if show_progress:
            ProgressManager.update_progress(min(step, total_steps))

        probabilities = np.abs(chosen_state) ** 2
        step += 1
        if show_progress:
            ProgressManager.update_progress(min(step, total_steps))

        state_prob_dict = {
            format(i, f"0{num_qubits}b"): probabilities[i] for i in range(hilbert_dim)
        }
        step += hilbert_dim
        if show_progress:
            ProgressManager.update_progress(total_steps)

    return state_prob_dict


def create_density_matrix(eigenvector, show_progress=False):
    """
    Constructs the density matrix from a given eigenvector.
    """
    with (
        ProgressManager.progress("Constructing Density Matrix", 1)
        if show_progress
        else ProgressManager.dummy_context()
    ):
        density_matrix = np.outer(eigenvector, np.conj(eigenvector))
        if show_progress:
            ProgressManager.update_progress(1)
    return density_matrix


def compute_reduced_density_matrix(density_matrix, configuration, show_progress=False):
    """
    Computes the reduced density matrix by tracing out sites marked as 0 in the configuration.
    """
    num_qubits = int(np.log2(density_matrix.shape[0]))
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

        reshaped_matrix = density_matrix.reshape([2] * (2 * num_qubits))
        step += 1
        if show_progress:
            ProgressManager.update_progress(min(step, total_steps))

        current_dim = num_qubits
        for atom in reversed(range(num_qubits)):
            if configuration[atom] == 0:
                reshaped_matrix = np.trace(
                    reshaped_matrix, axis1=atom, axis2=atom + current_dim
                )
                current_dim -= 1
                step += 1
                if show_progress:
                    ProgressManager.update_progress(min(step, total_steps))

        dim_subsystem = 2**subsystem_size
        reduced_density_matrix = reshaped_matrix.reshape((dim_subsystem, dim_subsystem))
        step += 1
        if show_progress:
            ProgressManager.update_progress(min(step, total_steps))

        end_time = time.time()
        if show_progress:
            print("\r" + " " * 80, end="")
            print(
                f"\rReduced Density Matrix computed in {end_time - start_time:.2f} seconds.",
                flush=True,
            )
            ProgressManager.update_progress(total_steps)

    return reduced_density_matrix
