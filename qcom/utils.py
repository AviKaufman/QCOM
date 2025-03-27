from .progress import ProgressManager
import numpy as np
import time
from scipy.sparse.linalg import eigsh

"""
Provides utility functions for the qcom package.
"""


def find_eigenstate(hamiltonian, state_index=0, show_progress=False):
    """
    Computes a specific eigenstate of the Hamiltonian efficiently.
    """
    if not isinstance(hamiltonian, np.ndarray):
        hamiltonian = hamiltonian.toarray()

    with (
        ProgressManager.progress("Finding Eigenstate", 1)
        if show_progress
        else ProgressManager.dummy_context()
    ):
        if show_progress:
            print(
                "\rFinding Eigenstate... This may take some time. Please wait.",
                end="",
                flush=True,
            )
        start_time = time.time()

        if state_index == 0:
            eigenvalues, eigenvectors = eigsh(hamiltonian, k=1, which="SA", tol=1e-10)
        else:
            eigenvalues, eigenvectors = eigsh(
                hamiltonian, k=state_index + 1, which="SA", tol=1e-10
            )

        chosen_eigenvalue = eigenvalues[state_index]
        chosen_eigenvector = eigenvectors[:, state_index]
        end_time = time.time()

        if show_progress:
            print("\r" + " " * 80, end="")
            print(
                f"\rEigenstate {state_index} found in {end_time - start_time:.2f} seconds.",
                flush=True,
            )
            ProgressManager.update_progress(1)

    return chosen_eigenvalue, chosen_eigenvector


def order_dict(inp_dict):
    """
    Orders a dictionary based on binary keys interpreted as integers.

    Args:
        inp_dict (dict): Dictionary where keys are binary strings.

    Returns:
        dict: Ordered dictionary sorted by integer values of binary keys.
    """
    ordered_items = sorted(inp_dict.items(), key=lambda item: int(item[0], 2))
    return dict(ordered_items)


def part_dict(inp_dict, indices):
    """
    Extracts a subset of bits from each binary string based on given indices.

    Args:
        inp_dict (dict): Dictionary where keys are binary strings.
        indices (list): List of indices specifying which bits to extract.

    Returns:
        dict: New dictionary where keys contain only the extracted bits.
    """
    new_dict = {}

    for key, value in inp_dict.items():
        extracted_bits = "".join(
            key[i] for i in indices
        )  # Extract only relevant indices
        if extracted_bits in new_dict:
            new_dict[extracted_bits] += value  # Sum probabilities for duplicates
        else:
            new_dict[extracted_bits] = value

    return new_dict
