import numpy as np
from scipy.linalg import eigh


def exact_diagonalization(model, atoms):
    """
    Build the Hamiltonian for `model` on `atoms`, diagonalize it,
    and return its eigenvalues and eigenvectors.
    """
    H = model.build(atoms, t=None)
    # eigh returns (eigvals, eigvecs)
    return eigh(H)
