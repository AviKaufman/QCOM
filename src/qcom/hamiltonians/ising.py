"""
Transverse-field Ising Hamiltonian builders.

Model
-----
H = - sum_{i<j} J_ij Z_i Z_j - sum_i hx_i X_i - sum_i hz_i Z_i

Basis convention follows the rest of QCOM: site 0 is the most significant bit.
For Pauli Z, bit 0 has eigenvalue +1 and bit 1 has eigenvalue -1.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Iterable, Mapping, cast

import numpy as np
import scipy.sparse as sp

from ..lattice_register import LatticeRegister
from .base import BaseHamiltonian

__all__ = ["IsingHamiltonian", "IsingParams", "build_ising"]


@dataclass(frozen=True)
class IsingParams:
    """Site and pair parameters for the transverse-field Ising model."""

    J: np.ndarray
    hx: np.ndarray
    hz: np.ndarray


def _as_site_array(x: float | Iterable[float] | None, n: int, name: str) -> np.ndarray:
    if x is None:
        arr = np.zeros(n, dtype=np.float64)
    elif isinstance(x, Real):
        arr = np.full(n, float(x), dtype=np.float64)
    else:
        values = cast(Iterable[float], x)
        arr = np.asarray(list(values), dtype=np.float64)
        if arr.ndim != 1 or arr.size != n:
            raise ValueError(f"{name} must be None, a scalar, or a length-{n} array.")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN/Inf.")
    return np.ascontiguousarray(arr, dtype=np.float64)


def _as_coupling_matrix(J: float | Iterable[float] | np.ndarray, n: int) -> np.ndarray:
    if isinstance(J, Real):
        mat = np.zeros((n, n), dtype=np.float64)
        for i in range(n - 1):
            mat[i, i + 1] = mat[i + 1, i] = float(J)
    else:
        arr = np.asarray(J, dtype=np.float64)
        if arr.ndim == 1:
            if arr.size != max(n - 1, 0):
                raise ValueError(f"1D J must have length N-1={max(n - 1, 0)}.")
            mat = np.zeros((n, n), dtype=np.float64)
            for i, value in enumerate(arr):
                mat[i, i + 1] = mat[i + 1, i] = float(value)
        elif arr.ndim == 2 and arr.shape == (n, n):
            mat = np.asarray(arr, dtype=np.float64).copy()
            mat = 0.5 * (mat + mat.T)
            np.fill_diagonal(mat, 0.0)
        else:
            raise ValueError("J must be a scalar, a length-N-1 array, or an NxN matrix.")
    if not np.isfinite(mat).all():
        raise ValueError("J contains NaN/Inf.")
    return np.ascontiguousarray(mat, dtype=np.float64)


class IsingHamiltonian(BaseHamiltonian):
    """
    Transverse-field Ising Hamiltonian.
    """

    def __init__(self, register: LatticeRegister, params: IsingParams):
        n = len(register)
        if params.J.shape != (n, n):
            raise ValueError("J must be shape (N, N).")
        if params.hx.shape != (n,) or params.hz.shape != (n,):
            raise ValueError("hx and hz must be shape (N,).")
        self.register = register
        self.params = params
        self._diag_cache: np.ndarray | None = None

    @classmethod
    def from_register(
        cls,
        register: LatticeRegister,
        *,
        J: float | Iterable[float] | np.ndarray,
        hx: float | Iterable[float] | None = None,
        hz: float | Iterable[float] | None = None,
    ) -> "IsingHamiltonian":
        n = len(register)
        if n == 0:
            raise ValueError("Register is empty.")
        params = IsingParams(
            J=_as_coupling_matrix(J, n),
            hx=_as_site_array(hx, n, "hx"),
            hz=_as_site_array(hz, n, "hz"),
        )
        return cls(register, params)

    @property
    def num_sites(self) -> int:
        return len(self.register)

    @property
    def hilbert_dim(self) -> int:
        return 1 << self.num_sites

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(np.float64)

    def parameters(self) -> Mapping[str, object]:
        return {
            "J_shape": self.params.J.shape,
            "hx_min": float(self.params.hx.min()),
            "hx_max": float(self.params.hx.max()),
            "hz_min": float(self.params.hz.min()),
            "hz_max": float(self.params.hz.max()),
        }

    def _diagonal(self) -> np.ndarray:
        if self._diag_cache is None:
            self._diag_cache = _ising_diagonal_from_bits(
                self.num_sites, self.params.J, self.params.hz
            )
        assert self._diag_cache is not None
        return self._diag_cache

    def _matvec(self, psi: np.ndarray) -> np.ndarray:
        psi = np.asarray(psi, dtype=np.float64, order="C")
        if psi.shape != (self.hilbert_dim,):
            raise ValueError(f"psi must have shape ({self.hilbert_dim},).")

        out = self._diagonal() * psi
        idx = np.arange(self.hilbert_dim, dtype=np.int64)
        for i, field in enumerate(self.params.hx):
            if field == 0.0:
                continue
            mask = 1 << (self.num_sites - 1 - i)
            out -= field * psi[idx ^ mask]
        return out

    def to_sparse(self) -> "sp.csr_matrix":
        n = self.num_sites
        dim = self.hilbert_dim
        rows_parts: list[np.ndarray] = []
        cols_parts: list[np.ndarray] = []
        data_parts: list[np.ndarray] = []

        idx = np.arange(dim, dtype=np.int64)
        rows_parts.append(idx)
        cols_parts.append(idx)
        data_parts.append(self._diagonal())

        for i, field in enumerate(self.params.hx):
            if field == 0.0:
                continue
            mask = 1 << (n - 1 - i)
            rows_parts.append(idx)
            cols_parts.append(idx ^ mask)
            data_parts.append(np.full(dim, -field, dtype=np.float64))

        rows = np.concatenate(rows_parts)
        cols = np.concatenate(cols_parts)
        data = np.concatenate(data_parts)
        return sp.coo_matrix((data, (rows, cols)), shape=(dim, dim), dtype=np.float64).tocsr()


def _ising_diagonal_from_bits(n: int, J: np.ndarray, hz: np.ndarray) -> np.ndarray:
    dim = 1 << n
    idx = np.arange(dim, dtype=np.int64)
    z_values: list[np.ndarray] = []
    for i in range(n):
        mask = 1 << (n - 1 - i)
        z_values.append(np.where((idx & mask) == 0, 1.0, -1.0))

    diag = np.zeros(dim, dtype=np.float64)
    for i in range(n):
        diag -= hz[i] * z_values[i]
        for j in range(i + 1, n):
            coupling = J[i, j]
            if coupling != 0.0:
                diag -= coupling * z_values[i] * z_values[j]
    return diag


def build_ising(
    register: LatticeRegister,
    *,
    J: float | Iterable[float] | np.ndarray,
    hx: float | Iterable[float] | None = None,
    hz: float | Iterable[float] | None = None,
) -> IsingHamiltonian:
    """Construct a transverse-field Ising Hamiltonian."""
    return IsingHamiltonian.from_register(register, J=J, hx=hx, hz=hz)
