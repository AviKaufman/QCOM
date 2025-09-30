from typing import Union, Callable, Optional
import numpy as np
from .base import HamiltonianModel


def _unwrap(param: Union[float, Callable[[float], float]], t: Optional[float]) -> float:
    """
    If `param` is a float, just return it.
    If it's a callable, call it at time t (defaulting to 0.0 if t is None).
    Always returns a float.
    """
    if callable(param):
        # ensure we give the function a real float
        t0: float = t if t is not None else 0.0
        return float(param(t0))
    else:
        return float(param)


def _op_on_site(op: np.ndarray, site: int, N: int) -> np.ndarray:
    """
    Construct the N-site operator that acts as `op` on site `site`
    and identity everywhere else, via Kronecker products.
    """
    result = np.array([1.0], dtype=complex)
    for i in range(N):
        result = np.kron(result, op if i == site else np.eye(2, dtype=complex))
    return result


class IsingParams(HamiltonianModel):
    """
    Transverse-field Ising model:
      H = -J ∑_{i=0..N-2} Z_i Z_{i+1}
          -h ∑_{i=0..N-1} X_i
    Optionally with periodic boundary conditions.

    Parameters
    ----------
    J : float or Callable[[float], float]
      Spin-spin coupling (constant or time-dependent J(t)).
    h : float or Callable[[float], float]
      Transverse field strength (constant or h(t)).
    pbc : bool
      If True, also include Z_{N-1} Z_0 coupling.
    """

    def __init__(self, J, h, pbc: bool = False):
        self.J = J
        self.h = h
        self.pbc = pbc

    def build(self, atoms, t=None):
        """
        Build the 2^Nx2^N Ising Hamiltonian for the given AtomRegister.

        Parameters
        ----------
        atoms : AtomRegister
            Your preconstructed register of N atom-sites.
        t : float, optional
            Time at which to evaluate J(t) and h(t). If None, uses constants.

        Returns
        -------
        H : np.ndarray, shape (2^N, 2^N)
        """
        # unwrap + force to float
        J_param = _unwrap(self.J, t)
        h_param = _unwrap(self.h, t)
        try:
            J = float(J_param)
            h = float(h_param)
        except (TypeError, ValueError):
            raise ValueError(
                f"IsingParams: J and h must evaluate to floats; got {J_param!r}, {h_param!r}"
            )

        N = len(atoms)
        dim = 2**N
        H = np.zeros((dim, dim), dtype=complex)

        # Pauli matrices
        sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

        # spin–spin coupling: -J ∑ Z_i Z_{i+1}
        for i in range(N - 1):
            Zi = _op_on_site(sigma_z, i, N)
            Zj = _op_on_site(sigma_z, i + 1, N)
            H -= J * (Zi @ Zj)

        if self.pbc and N > 1:
            ZN_1 = _op_on_site(sigma_z, N - 1, N)
            Z0 = _op_on_site(sigma_z, 0, N)
            H -= J * (ZN_1 @ Z0)

        # transverse field: -h ∑ X_i
        for i in range(N):
            Xi = _op_on_site(sigma_x, i, N)
            H -= h * Xi

        return H
