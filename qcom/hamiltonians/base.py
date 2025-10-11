# qcom/hamiltonians/base.py
"""
Abstract base class for Hamiltonians in QCOM.

This module defines `BaseHamiltonian`, a lightweight builder interface that
decouples Hamiltonian *construction* from its *representation*. Subclasses may
implement either a fast `_matvec` (preferred for very large Hilbert spaces) or
a sparse materializer `to_sparse()` (preferred when an explicit sparse matrix
is compact and convenient). High-level materializers (`to_linear_operator`,
`to_sparse`, `to_dense`) and convenience ops (`apply`, `ground_state`) are
provided on top.

Design highlights
-----------------
• Minimal contract: `num_sites`, `hilbert_dim`, `dtype`, and EITHER `_matvec` or
  `to_sparse`. Implement both if you can.
• Lazy SciPy usage: we import `scipy.sparse.linalg` only when needed and raise
  a clear error if SciPy isn’t installed.
• Memory-savvy defaults: prefer `_matvec` → `LinearOperator`; fall back to
  sparse if present; build dense only on explicit request (small systems).

Typical subclassing pattern
---------------------------
class MyH(BaseHamiltonian):
    @property
    def num_sites(self): ...
    @property
    def hilbert_dim(self): ...
    @property
    def dtype(self): return np.float64  # or np.complex128

    # Option A (recommended for big systems):
    def _matvec(self, psi): ...  # return H @ psi

    # Option B (if sparse assembly is natural):
    def to_sparse(self): ...  # return a scipy.sparse.spmatrix

API synopsis
------------
- to_linear_operator()  -> scipy.sparse.linalg.LinearOperator
- to_sparse()           -> scipy.sparse.spmatrix (override in subclass)
- to_dense()            -> np.ndarray
- apply(psi)            -> H @ psi  (without building dense)
- ground_state(...)     -> extremal eigenpairs via SciPy (eigsh/eigs)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, Optional
import numpy as np


# -------------------- Base Class --------------------

class BaseHamiltonian(ABC):
    """
    Abstract interface for Hamiltonians in QCOM.

    Subclasses must provide:
      - `num_sites` (int)
      - `hilbert_dim` (int)
      - `dtype` (np.dtype)
    And EITHER:
      - `_matvec(self, psi) -> np.ndarray`
      - `to_sparse(self) -> "sp.spmatrix"`
    """

    # -------------------- Minimal identity --------------------
    @property
    @abstractmethod
    def num_sites(self) -> int:
        """Number of lattice sites/spins represented."""

    @property
    @abstractmethod
    def hilbert_dim(self) -> int:
        """Dimension of the represented Hilbert space."""

    @property
    @abstractmethod
    def dtype(self) -> np.dtype:
        """Dtype of H's matrix elements (e.g., np.float64 or np.complex128)."""

    # Hermiticity is the default; override if needed (e.g., non-Hermitian models).
    is_hermitian: bool = True

    # -------------------- Optional descriptor --------------------
    def parameters(self) -> Mapping[str, Any]:
        """Optional summary of construction parameters (for logging/debug)."""
        return {}

    # -------------------- Backends (subclasses override at least one) --------------------
    def to_sparse(self):
        """Return a scipy.sparse.spmatrix representation of H, if implemented."""
        raise NotImplementedError("to_sparse() not implemented for this Hamiltonian.")

    def _matvec(self, psi: np.ndarray) -> np.ndarray:
        """Multiply H @ psi without materializing a dense matrix."""
        raise NotImplementedError("_matvec() not implemented for this Hamiltonian.")

    # -------------------- Materializers --------------------
    def to_linear_operator(self):
        """
        Build a scipy.sparse.linalg.LinearOperator for H.

        Preference order:
          1) If `_matvec` is overridden by the subclass, wrap it directly.
          2) Otherwise, wrap `to_sparse()` (if implemented) as matvec.
        """
        sp_linalg = _require_scipy_linalg("to_linear_operator")
        shape = (self.hilbert_dim, self.hilbert_dim)
        dtype = self.dtype

        if _has_custom_matvec(self):
            def mv(x):
                x = np.asarray(x, dtype=dtype, order="C")
                return self._matvec(x)
            return sp_linalg.LinearOperator(shape=shape, matvec=mv, dtype=dtype)

        try:
            sp = self.to_sparse()
        except NotImplementedError as e:
            raise RuntimeError(
                "Neither _matvec nor to_sparse is implemented; cannot form a LinearOperator."
            ) from e

        def mv_sparse(x):
            return sp @ x
        return sp_linalg.LinearOperator(shape=shape, matvec=mv_sparse, dtype=dtype)

    def to_dense(self) -> np.ndarray:
        """
        Materialize a dense ndarray for H.

        Strategy:
          1) If `to_sparse()` is implemented and SciPy is available, return `to_sparse().toarray()`.
          2) Else build from LinearOperator by applying to identity columns (O(d^2) work).
             Suitable only for small Hilbert dimensions.
        """
        try:
            sp = self.to_sparse()
            return sp.toarray()
        except NotImplementedError:
            pass
        except Exception:
            raise

        H = self.to_linear_operator()
        d = self.hilbert_dim
        out = np.empty((d, d), dtype=self.dtype)
        eye = np.eye(d, dtype=self.dtype)
        for j in range(d):
            out[:, j] = H @ eye[:, j]
        return out

    # -------------------- Convenience ops --------------------
    def apply(self, psi: np.ndarray) -> np.ndarray:
        """
        Compute H @ psi without materializing dense matrices.

        Uses `_matvec` if implemented; else wraps `to_sparse()`.
        """
        psi = np.asarray(psi, dtype=self.dtype, order="C")
        if psi.shape[0] != self.hilbert_dim:
            raise ValueError(
                f"apply: psi has incompatible shape {psi.shape}, "
                f"expected (hilbert_dim,) with hilbert_dim={self.hilbert_dim}."
            )
        if _has_custom_matvec(self):
            return self._matvec(psi)
        sp = self.to_sparse()
        return sp @ psi

    def ground_state(
        self,
        k: int = 1,
        which: str = "SA",
        maxiter: Optional[int] = None,
        tol: float = 0.0,
        return_eigenvectors: bool = True,
    ):
        """
        Compute extremal eigenpairs using SciPy ARPACK wrappers.

        Parameters
        ----------
        k : int
            Number of extremal eigenvalues to compute (default 1).
            Must satisfy 1 <= k < hilbert_dim.
        which : str
            For Hermitian problems (eigsh): 'SA', 'LA', etc.
            For non-Hermitian problems (eigs): 'SR', 'LR', etc.
        maxiter : Optional[int]
            Max iterations for the solver.
        tol : float
            Convergence tolerance.
        return_eigenvectors : bool
            If False, return only eigenvalues.

        Returns
        -------
        evals  (and evecs if requested)
        """
        d = self.hilbert_dim
        if not (1 <= k < d):
            raise ValueError(f"ground_state: k must satisfy 1 <= k < hilbert_dim (k={k}, hilbert_dim={d}).")

        if self.is_hermitian:
            sp_linalg = _require_scipy_linalg("ground_state (Hermitian)")
            try:
                sp = self.to_sparse()
                evals, evecs = sp_linalg.eigsh(sp, k=k, which=which, maxiter=maxiter, tol=tol)
            except NotImplementedError:
                Lop = self.to_linear_operator()
                evals, evecs = sp_linalg.eigsh(Lop, k=k, which=which, maxiter=maxiter, tol=tol)
        else:
            sp_linalg = _require_scipy_linalg("ground_state (non-Hermitian)")
            try:
                sp = self.to_sparse()
                evals, evecs = sp_linalg.eigs(sp, k=k, which=which, maxiter=maxiter, tol=tol)
            except NotImplementedError:
                Lop = self.to_linear_operator()
                evals, evecs = sp_linalg.eigs(Lop, k=k, which=which, maxiter=maxiter, tol=tol)

        if return_eigenvectors:
            return evals, evecs
        return evals

    # -------------------- Niceties --------------------
    def __repr__(self) -> str:
        cls = self.__class__.__name__
        try:
            p = dict(self.parameters())
        except Exception:
            p = {}
        summary = ", ".join(f"{k}={v}" for k, v in p.items()) if p else "…"
        return f"{cls}(N={self.num_sites}, dim={self.hilbert_dim}, dtype={self.dtype}, params={{ {summary} }})"


# -------------------- Internal helpers --------------------

def _require_scipy_linalg(where: str):
    """Import scipy.sparse.linalg lazily with a clear error if missing."""
    try:
        import scipy.sparse.linalg as sp_linalg  # type: ignore
        return sp_linalg
    except Exception as e:
        raise RuntimeError(
            f"{where} requires SciPy. Please install with `pip install scipy`."
        ) from e


def _has_custom_matvec(obj: BaseHamiltonian) -> bool:
    """Return True if `obj._matvec` is overridden by the subclass."""
    return getattr(type(obj), "_matvec") is not BaseHamiltonian._matvec