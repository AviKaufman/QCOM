# qcom/hamiltonians/rydberg.py
"""
Rydberg Hamiltonian Builders

Purpose
-------
Provide an implementation of the standard two-level Rydberg Hamiltonian
suitable for small-scale exact diagonalization or sparse matrix simulations.
This file defines the `RydbergHamiltonian` object and convenience builders.

Core Model
----------
The Hamiltonian follows the standard form:

    H = sum_i (Ω_i/2)[cos φ_i σ^x_i + sin φ_i σ^y_i]
        - sum_i Δ_i n_i
        + sum_{i<j} V_{ij} n_i n_j

with:
    n_i = (I - σ^z_i)/2
    V_{ij} = C6 / r_ij^6

Inputs are taken from a `LatticeRegister` (positions in meters),
together with site-dependent or global values of Ω, Δ, and φ.

Key Guarantees
--------------
• Positions: drawn from `LatticeRegister`, always in SI meters.
• Ω (Rabi), Δ (detuning), φ (phase) can be scalars or arrays (broadcast to sites).
• C6: supplied in rad·s⁻¹·m⁶ (ensures consistency across species).
• Dense and sparse backends supported (CSR via SciPy).
• Hermitian by construction, dtype auto-promotes to complex when required.

Design Philosophy
-----------------
• Object-first workflow: construct a `RydbergHamiltonian` instance containing
  the register, parameters, and precomputed interaction matrix. The object can
  be materialized into different backends (dense array, sparse matrix) on demand.

• Backends:
  - `to_dense()`: exact matrix (2^N × 2^N), useful for ED and debugging.
  - `to_sparse()`: CSR matrix; still exponential in size but more memory efficient.
    Uses a flip-table build for the transverse drive to avoid unnecessary kron work.

• Broadcasting: scalars automatically expand to arrays for site dependence,
  keeping simple use cases concise while preserving flexibility.

Future Extensions (non-breaking)
--------------------------------
• Support for time dependence via `TimeSeries`.
• Optional constraints (e.g., blockade Hilbert-space reduction).
• Additional terms (e.g., longitudinal fields, disorder).
• GPU-backed backends and tensor-network interfaces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

# -------------------- Local imports --------------------
from ..lattice_register import LatticeRegister

from .base import BaseHamiltonian

# For type-checking only (avoids importing SciPy at runtime for annotations)
from typing import TYPE_CHECKING
if TYPE_CHECKING:  # pragma: no cover
    from scipy.sparse import csr_matrix

import scipy.sparse as sp  # type: ignore


# -------------------- Utilities --------------------

def _as_1d_float_array(x: float | Iterable[float], n: int, name: str) -> np.ndarray:
    """
    Coerce a scalar or iterable to a 1D float64 array of length n.

    Raises:
        ValueError: if the array cannot be broadcast to length n or contains NaN/Inf.
    """
    if np.isscalar(x):
        arr = np.full(n, float(x), dtype=np.float64)
    else:
        arr = np.asarray(list(x), dtype=np.float64)
        if arr.ndim != 1 or arr.size != n:
            raise ValueError(f"{name} must be a scalar or length-{n} array")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN/Inf")
    return arr


def _needs_complex(phi: np.ndarray) -> bool:
    """
    Determine if phases imply a complex Hamiltonian via σ^y contributions.
    """
    # Any sin(phi) != 0 introduces ±i in σ^y term -> complex dtype needed.
    return np.any(np.sin(phi) != 0.0)


# Pauli matrices (2x2) as NumPy arrays (dense building uses these)
_SIGMA_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
_SIGMA_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
_SIGMA_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.float64)
_I2      = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)

# For sparse building, keep 2x2 blocks as CSR; lazily defined if SciPy present
def _sp_blocks():  # pragma: no cover - trivial
    if sp is None:
        raise RuntimeError("SciPy is required for sparse Rydberg builders (pip install scipy).")
    sx = sp.csr_matrix(_SIGMA_X)
    sy = sp.csr_matrix(_SIGMA_Y)
    sz = sp.csr_matrix(_SIGMA_Z)
    ii = sp.eye(2, dtype=np.float64, format="csr")
    return sx, sy, sz, ii


# -------------------- Data Object --------------------

@dataclass(frozen=True)
class RydbergParams:
    """
    Immutable container for site-dependent parameters and interactions.

    Attributes:
        omega: (N,) Rabi frequencies Ω_i (rad/s)
        delta: (N,) Detunings Δ_i (rad/s)
        phi:   (N,) Phases φ_i (radians)
        vij:   (N,N) Pairwise interaction matrix V_ij (rad/s)
    """
    omega: np.ndarray
    delta: np.ndarray
    phi:   np.ndarray
    vij:   np.ndarray


# -------------------- Main Class --------------------

class RydbergHamiltonian(BaseHamiltonian):
    """
    Object-first Rydberg Hamiltonian.

    Holds the register, per-site parameters (Ω, Δ, φ), and precomputed
    interactions V_ij = C6 / r_ij^6 (with zeros on the diagonal).
    Provides materialization methods to build dense or sparse matrices.
    """

    # -------------------- Construction --------------------
    def __init__(self, register: LatticeRegister, params: RydbergParams):
        if len(register) != params.omega.size:
            raise ValueError("Parameter length mismatch with register size")
        if params.vij.shape != (len(register), len(register)):
            raise ValueError("vij must be shape (N, N)")
        self.register = register
        self.params = params

    @classmethod
    def from_register(
        cls,
        register: LatticeRegister,
        *,
        C6: float,
        Omega: float | Iterable[float],
        Delta: float | Iterable[float],
        Phi: float | Iterable[float] = 0.0,
        cutoff: Optional[float] = None,
    ) -> "RydbergHamiltonian":
        """
        Build a `RydbergHamiltonian` by broadcasting per-site parameters and
        precomputing interactions from positions.
        """
        N = len(register)
        if N == 0:
            raise ValueError("Register is empty")

        omega = _as_1d_float_array(Omega, N, "Omega")
        delta = _as_1d_float_array(Delta, N, "Delta")
        phi   = _as_1d_float_array(Phi,   N, "Phi")

        # Pairwise distances (meters)
        D = register.distances()  # (N, N), zeros on the diagonal by construction

        # Interaction matrix V_ij = C6 / r^6, with V_ii = 0.
        with np.errstate(divide="ignore"):
            r6 = np.where(D > 0.0, D**6, np.inf)  # avoid div-by-zero on diagonal
            vij = C6 / r6
        np.fill_diagonal(vij, 0.0)

        # Optional cutoff
        if cutoff is not None:
            cutoff = float(cutoff)
            if cutoff <= 0:
                raise ValueError("cutoff must be positive (meters)")
            vij = np.where(D <= cutoff, vij, 0.0)

        params = RydbergParams(
            omega=np.ascontiguousarray(omega, dtype=np.float64),
            delta=np.ascontiguousarray(delta, dtype=np.float64),
            phi=np.ascontiguousarray(phi, dtype=np.float64),
            vij=np.ascontiguousarray(vij, dtype=np.float64),
        )
        return cls(register, params)

    # -------------------- Introspection --------------------
    @property
    def num_sites(self) -> int:
        """Number of sites."""
        return len(self.register)

    @property
    def hilbert_dim(self) -> int:
        """Dimension of the computational basis (2^N)."""
        return 1 << self.num_sites

    @property
    def dtype(self) -> np.dtype:
        # real if all sin(phi_i)=0, else complex
        return np.complex128 if _needs_complex(self.params.phi) else np.float64

    def __repr__(self) -> str:
        return (
            f"RydbergHamiltonian(N={self.num_sites}, "
            f"Omega∈[{self.params.omega.min():.3e},{self.params.omega.max():.3e}], "
            f"Delta∈[{self.params.delta.min():.3e},{self.params.delta.max():.3e}], "
            f"phi∈[{self.params.phi.min():.3e},{self.params.phi.max():.3e}])"
        )

    # -------------------- Dense Backend --------------------
    def to_dense(self) -> np.ndarray:
        """
        Materialize H as a dense (2^N × 2^N) NumPy array.

        Returns:
            np.ndarray: Hermitian matrix. dtype is float64 if sin(φ_i)=0 ∀i, else complex128.

        Notes:
            • Scales as O(2^N) memory and O(N·2^N) time.
            • For larger N, consider `to_sparse()` or matvec-based solvers.
        """
        N = self.num_sites
        if N == 0:
            return np.zeros((1, 1), dtype=np.float64)

        # Decide dtype based on phases (σ^y introduces ±i)
        use_complex = _needs_complex(self.params.phi)
        dtype = np.complex128 if use_complex else np.float64

        dim = 1 << N
        H = np.zeros((dim, dim), dtype=dtype)

        # Pre-build single-site operator cache to avoid repeated kron chains
        x_ops = []
        n_ops = []
        for i in range(N):
            x_ops.append(_kron_local_dense(N, i, _SIGMA_X, dtype))
            n_i = 0.5 * (_kron_local_dense(N, i, _I2, dtype) - _kron_local_dense(N, i, _SIGMA_Z, dtype))
            n_ops.append(n_i)

        y_ops = None
        if use_complex:
            y_ops = []
            for i in range(N):
                y_ops.append(_kron_local_dense(N, i, _SIGMA_Y, dtype))

        # Transverse drive: sum_i (Ω_i/2)[cos φ_i σx_i + sin φ_i σy_i]
        for i in range(N):
            ci = np.cos(self.params.phi[i])
            si = np.sin(self.params.phi[i])
            term = (self.params.omega[i] / 2.0) * (ci * x_ops[i])
            if use_complex and si != 0.0:
                term = term + (self.params.omega[i] / 2.0) * (si * y_ops[i])
            H += term

        # Detuning: - sum_i Δ_i n_i
        for i in range(N):
            H -= self.params.delta[i] * n_ops[i]

        # Interactions: sum_{i<j} V_ij n_i n_j
        for i in range(N):
            ni = n_ops[i]
            for j in range(i + 1, N):
                vij = self.params.vij[i, j]
                if vij == 0.0:
                    continue
                H += vij * (ni @ n_ops[j])

        # If complex dtype was used but the result is numerically real, safely downcast.
        if use_complex:
            imax = float(np.max(np.abs(H.imag)))
            rmax = float(np.max(np.abs(H.real)))
            tol = 1e-12 * max(1.0, rmax)
            if imax <= tol:
                return H.real.astype(np.float64, copy=False)

        return H

    # -------------------- Sparse Backend (flip-table build) --------------------
    def to_sparse(self) -> "sp.csr_matrix":
        """
        Materialize H as a sparse CSR matrix (SciPy), built via bit-flip (flip table)
        and diagonal accumulation.

        Returns:
            scipy.sparse.csr_matrix: Hermitian matrix in CSR format.

        Raises:
            RuntimeError: if SciPy is not available.

        Notes:
            • Still exponential in dimension (2^N), but only true nonzeros are generated.
            • Drive term: O(N·2^N) off-diagonal nonzeros via single-bit flips.
            • Diagonals (detuning + interactions): O(2^N) entries.
        """
        if sp is None:
            raise RuntimeError("SciPy is required for sparse Rydberg builders (pip install scipy).")

        N = self.num_sites
        if N == 0:
            return sp.csr_matrix((1, 1), dtype=np.float64)

        # dtype selection: any σ^y contribution forces complex
        use_complex = _needs_complex(self.params.phi)
        dtype = np.complex128 if use_complex else np.float64

        dim = 1 << N
        rows_parts: list[np.ndarray] = []
        cols_parts: list[np.ndarray] = []
        data_parts: list[np.ndarray] = []

        # Drive term via one-bit flips
        r, c, d, dtype_drive = _drive_coo_from_bitflips(N, self.params.omega, self.params.phi)
        if np.dtype(dtype_drive) != np.dtype(dtype):
            d = d.astype(dtype, copy=False)
        rows_parts.append(r); cols_parts.append(c); data_parts.append(d)

        # Detuning diagonal: -Σ_i Δ_i n_i
        diag = _detuning_diagonal_from_bits(N, self.params.delta, dtype)

        # Interaction diagonal: Σ_{i<j} V_ij n_i n_j
        if np.any(self.params.vij):
            diag += _interaction_diagonal_from_bits(N, self.params.vij, dtype)

        # Add diagonal triplets
        idx = np.arange(dim, dtype=np.int64)
        rows_parts.append(idx); cols_parts.append(idx); data_parts.append(diag)

        # Assemble COO -> CSR
        R = np.concatenate(rows_parts)
        C = np.concatenate(cols_parts)
        D = np.concatenate(data_parts).astype(dtype, copy=False)
        H = sp.coo_matrix((D, (R, C)), shape=(dim, dim), dtype=dtype).tocsr()

        # If complex but numerically real, downcast safely.
        if H.dtype == np.complex128:
            imax = float(np.max(np.abs(H.data.imag))) if H.data.size else 0.0
            rmax = float(np.max(np.abs(H.data.real))) if H.data.size else 0.0
            tol = 1e-12 * max(1.0, rmax)
            if imax <= tol:
                H = H.real.astype(np.float64, copy=False)

        return H


# -------------------- HELPERS: Flip-Table Sparse Build --------------------

def _drive_coo_from_bitflips(
    N: int,
    omega: np.ndarray,
    phi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.dtype]:
    """
    Build the drive term (Σ_i (Ω_i/2)[cos φ_i σ^x_i + sin φ_i σ^y_i]) directly
    as COO triplets via one-bit flips.

    Convention: MSB=0 (site 0 ↔ leftmost bit ↔ bit position N-1).
    For each site i, flipping that bit connects basis states m ↔ n=m^(1<<(N-1-i)).

    Upper-triangle entry (m<n): (Ω_i/2) e^{-i φ_i}
    Lower-triangle entry (n>m): (Ω_i/2) e^{+i φ_i}
    (Hermiticity is explicit.)

    Returns:
        rows, cols, data, dtype_of_data
    """
    dim = 1 << N
    rows = []
    cols = []
    data = []

    any_complex = np.any(np.sin(phi) != 0.0)
    all_states = np.arange(dim, dtype=np.int64)

    for i in range(N):
        mask = 1 << (N - 1 - i)  # MSB=0 mapping

        # only generate each unordered pair once: choose 'm' with bit i == 0
        m = all_states[(all_states & mask) == 0]
        n = m ^ mask

        ci = np.cos(phi[i])
        si = np.sin(phi[i])
        up_val  = (omega[i] * 0.5) * (ci - 1j * si)  # m→n
        low_val = (omega[i] * 0.5) * (ci + 1j * si)  # n→m

        rows.append(m); cols.append(n); data.append(np.full(m.size, up_val,  dtype=np.complex128))
        rows.append(n); cols.append(m); data.append(np.full(m.size, low_val, dtype=np.complex128))

    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    data = np.concatenate(data)

    if not any_complex:
        data = data.real.astype(np.float64, copy=False)
        return rows, cols, data, np.float64

    return rows, cols, data, np.complex128


def _detuning_diagonal_from_bits(
    N: int,
    delta: np.ndarray,
    dtype: np.dtype,
) -> np.ndarray:
    """
    Diagonal array for -Σ_i Δ_i n_i with n_i(b) = b_i (bit i occupancy).
    MSB=0 → bit position for site i is (N-1-i).
    """
    dim = 1 << N
    diag = np.zeros(dim, dtype=dtype)
    idx = np.arange(dim, dtype=np.int64)
    for i in range(N):
        mask = 1 << (N - 1 - i)
        diag -= delta[i] * ((idx & mask) != 0)
    return diag


def _interaction_diagonal_from_bits(
    N: int,
    vij: np.ndarray,
    dtype: np.dtype,
) -> np.ndarray:
    """
    Diagonal array for Σ_{i<j} V_ij n_i n_j with n_i(b) = b_i.
    Vectorized over basis states using bit tests. MSB=0 convention.
    """
    dim = 1 << N
    add_int = np.zeros(dim, dtype=dtype)
    idx = np.arange(dim, dtype=np.int64)

    # Precompute bit-occupancy booleans once
    bits_bool = []
    for i in range(N):
        mask = 1 << (N - 1 - i)
        bits_bool.append((idx & mask) != 0)

    for i in range(N):
        bi = bits_bool[i]
        if not np.any(bi):
            continue
        for j in range(i + 1, N):
            v = vij[i, j]
            if v == 0.0:
                continue
            add_int += v * (bi & bits_bool[j])
    return add_int


# -------------------- Kron Helpers (Dense) --------------------

def _kron_local_dense(N: int, which: int, op2: np.ndarray, dtype) -> np.ndarray:
    """
    Place a 2×2 operator `op2` on site `which` among N sites (0-indexed),
    returning a dense 2^N × 2^N matrix.
    """
    out = None
    for site in range(N):
        block = op2 if site == which else _I2
        out = block if out is None else np.kron(out, block)
    return out.astype(dtype, copy=False)


# -------------------- (Legacy) Kron Helpers (Sparse) --------------------
# Not used in the flip-table sparse builder, but kept for potential future use.

def _kron_local_sparse(N: int, which: int, op2: "sp.csr_matrix") -> "sp.csr_matrix":
    """
    Place a 2×2 sparse operator `op2` on site `which` among N sites (0-indexed),
    returning a CSR 2^N × 2^N matrix.
    """
    if sp is None:  # pragma: no cover
        raise RuntimeError("SciPy is required for sparse Rydberg builders (pip install scipy).")
    _, _, _, ii = _sp_blocks()
    out = None
    for site in range(N):
        block = op2 if site == which else ii
        out = block if out is None else sp.kron(out, block, format="csr")
    return out


# -------------------- Public Builder --------------------

def build_rydberg(
    register: LatticeRegister,
    *,
    C6: float,
    Omega: float | Iterable[float],
    Delta: float | Iterable[float],
    Phi: float | Iterable[float] = 0.0,
    cutoff: Optional[float] = None,
) -> RydbergHamiltonian:
    """
    Convenience function to construct a `RydbergHamiltonian` from inputs.
    """
    return RydbergHamiltonian.from_register(
        register, C6=C6, Omega=Omega, Delta=Delta, Phi=Phi, cutoff=cutoff
    )