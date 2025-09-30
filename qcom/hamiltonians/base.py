from abc import ABC, abstractmethod
import numpy as np
from typing import Optional


class HamiltonianModel(ABC):
    """
    Abstract base class for all Hamiltonian models.
    Requires subclasses to implement `build(atoms, t=None)` which
    returns the Hamiltonian matrix for the given AtomRegister.
    """

    @abstractmethod
    def build(self, atoms, t: Optional[float] = None) -> np.ndarray:
        """
        Construct and return the Hamiltonian matrix.

        Parameters
        ----------
        atoms : AtomRegister
            The register of atom positions.
        t : float, optional
            Time at which to evaluate any time-dependent parameters.
            If None, should use time-independent values.

        Returns
        -------
        H : np.ndarray, shape (2^N, 2^N)
            The Hamiltonian matrix.
        """
        ...
