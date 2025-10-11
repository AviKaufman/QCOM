from .._internal.progress import ProgressManager
import time
import numpy as np
from scipy.sparse.linalg import eigsh

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