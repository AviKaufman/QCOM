"""
qcom.hamiltonians.ising (stub)
==============================

Temporary stub so that imports like

    from qcom.hamiltonians import IsingHamiltonian, build_ising

work while the real Ising implementation is in progress.

Exports
-------
- IsingHamiltonian: a placeholder class matching the BaseHamiltonian interface.
- build_ising(...): convenience constructor returning the placeholder.

All materialization methods currently raise NotImplementedError.
"""

from __future__ import annotations

from typing import Any
import warnings
import numpy as np

from .base import BaseHamiltonian           # abstract interface
from ..lattice_register import LatticeRegister

__all__ = ["IsingHamiltonian", "build_ising"]


class IsingHamiltonian(BaseHamiltonian):
    """
    Placeholder for a transverse-field Ising Hamiltonian.

    Parameters
    ----------
    register : LatticeRegister
        Spin/atom layout (sites and geometry).
    J : float | np.ndarray
        Exchange coupling(s).
    hx : float | np.ndarray | None
        Transverse field(s) along x.
    hz : float | np.ndarray | None
        Longitudinal field(s) along z.
    """

    register: LatticeRegister
    J: float | np.ndarray
    hx: float | np.ndarray | None = None
    hz: float | np.ndarray | None = None

    # -------------------- Construction helpers --------------------

    @classmethod
    def from_register(
        cls,
        register: LatticeRegister,
        *,
        J: float | np.ndarray,
        hx: float | np.ndarray | None = None,
        hz: float | np.ndarray | None = None,
        **_: Any,
    ) -> "IsingHamiltonian":
        warnings.warn(
            "IsingHamiltonian is a stub. Materialization methods are not implemented yet.",
            RuntimeWarning,
            stacklevel=2,
        )
        return cls(register=register, J=J, hx=hx, hz=hz)

    # -------------------- BaseHamiltonian interface (stubbed) --------------------

    def to_sparse(self):
        raise NotImplementedError("IsingHamiltonian.to_sparse is not implemented yet.")

    def to_dense(self) -> np.ndarray:
        raise NotImplementedError("IsingHamiltonian.to_dense is not implemented yet.")

    def to_linear_operator(self):
        raise NotImplementedError("IsingHamiltonian.to_linear_operator is not implemented yet.")


# -------------------- Public convenience builder --------------------

def build_ising(
    register: LatticeRegister,
    *,
    J: float | np.ndarray,
    hx: float | np.ndarray | None = None,
    hz: float | np.ndarray | None = None,
    **kwargs: Any,
) -> IsingHamiltonian:
    """
    Convenience stub matching the eventual API. Returns the placeholder class.

    Notes
    -----
    This exists only to keep the public import surface stable while
    the full implementation is under development.
    """
    return IsingHamiltonian.from_register(register, J=J, hx=hx, hz=hz, **kwargs)