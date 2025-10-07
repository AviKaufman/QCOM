# qcom/hamiltonians/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, Optional
import numpy as np


class BaseHamiltonian(ABC):
    """
    Abstract interface for Hamiltonians in QCOM.

    Design goals
    ------------
    - Keep the *builder* object lightweight, independent of the concrete matrix form.
    - Allow subclasses to implement *either* `_matvec` (preferred for very large Hilbert spaces)
      or `to_sparse()` (preferred when you can assemble a compact sparse matrix).
    - Provide materializers:
        * `.to_linear_operator()`  -> scipy.sparse.linalg.LinearOperator
        * `.to_sparse()`           -> scipy.sparse.spmatrix (if implemented by subclass)
        * `.to_dense()`            -> np.ndarray (via sparse or linear operator)
      All SciPy usage is imported lazily with clear error messages if SciPy is missing.

    Required contract for subclasses
    --------------------------------
    - `num_sites`: number of lattice sites/spins/qudits.
    - `hilbert_dim`: total Hilbert space dimension for the representation used
      (e.g., 2**N for qubits, unless a symmetry-reduced space is used).
    - `dtype`: numpy dtype of matrix elements (typically np.float64 or np.complex128).
    - Either:
        * implement `_matvec(self, psi: np.ndarray) -> np.ndarray`, OR
        * implement `to_sparse(self) -> "sp.spmatrix"`.
      You may implement both for efficiency.

    Optional (recommended)
    ----------------------
    - `parameters() -> Mapping[str, Any]`: structured summary of key builder inputs.
    - `is_hermitian`: bool flag (defaults True).

    Notes
    -----
    - `.apply(psi)` multiplies by the Hamiltonian without materializing dense matrices.
    - `.ground_state()` is provided as a convenience and uses SciPy eigensolvers if available.
      It will prefer the sparse backend when present.
    """

    # ---------- Minimal identity ----------
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

    # Hermiticity is the default; override if needed (e.g., non-Hermitian effective models).
    is_hermitian: bool = True

    # ---------- Optional descriptor ----------
    def parameters(self) -> Mapping[str, Any]:
        """Optional summary of construction parameters (for logging/debug)."""
        return {}

    # ---------- Backends to optionally implement ----------
    def to_sparse(self):
        """
        Return a scipy.sparse.spmatrix representation of H, if implemented.

        Subclasses may override. The default raises NotImplementedError.
        """
        raise NotImplementedError("to_sparse() not implemented for this Hamiltonian.")

    def _matvec(self, psi: np.ndarray) -> np.ndarray:
        """
        Multiply H @ psi without materializing a dense matrix.

        Subclasses may override. The default raises NotImplementedError.
        """
        raise NotImplementedError("_matvec() not implemented for this Hamiltonian.")

    # ---------- Materializers built on top of the above ----------
    def to_linear_operator(self):
        """
        Build a scipy.sparse.linalg.LinearOperator for H.

        Prefers `_matvec` if available; otherwise, wraps `to_sparse().dot`.
        """
        sp_linalg = _require_scipy_linalg("to_linear_operator")
        shape = (self.hilbert_dim, self.hilbert_dim)
        dtype = self.dtype

        # If subclass implements _matvec, wrap it directly (most memory-efficient).
        try:
            _ = self._matvec  # will raise if not overridden
            def mv(x):
                x = np.asarray(x, dtype=dtype, order="C")
                return self._matvec(x)
            return sp_linalg.LinearOperator(shape=shape, matvec=mv, dtype=dtype)
        except NotImplementedError:
            pass

        # Fallback: use sparse, if available.
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
             This is only suitable for small Hilbert dimensions.
        """
        # Try sparse first for both speed and memory.
        try:
            sp = self.to_sparse()
            return sp.toarray()
        except NotImplementedError:
            pass
        except Exception:
            # If something else went wrong in to_sparse, surface it
            raise

        # Fall back to linear-operator build (expensive).
        H = self.to_linear_operator()
        d = self.hilbert_dim
        out = np.empty((d, d), dtype=self.dtype)
        eye = np.eye(d, dtype=self.dtype)
        # Column-by-column application
        for j in range(d):
            out[:, j] = H @ eye[:, j]
        return out

    # ---------- Convenience operations ----------
    def apply(self, psi: np.ndarray) -> np.ndarray:
        """
        Compute H @ psi without materializing dense matrices.

        Uses `_matvec` if implemented, else wraps `to_sparse()`.
        """
        psi = np.asarray(psi, dtype=self.dtype, order="C")
        if psi.shape[0] != self.hilbert_dim:
            raise ValueError(f"apply: psi has incompatible shape {psi.shape}, "
                             f"expected (hilbert_dim,) with hilbert_dim={self.hilbert_dim}.")
        try:
            return self._matvec(psi)
        except NotImplementedError:
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
        Compute lowest eigenvalues/eigenvectors using SciPy (if available).

        Parameters
        ----------
        k : int
            Number of extremal eigenvalues to compute (default 1).
        which : str
            'SA' (smallest algebraic), 'LA' (largest algebraic), etc.
            For Hermitian problems prefer eigsh semantics.
        maxiter : Optional[int]
            Max iterations for the solver.
        tol : float
            Convergence tolerance.
        return_eigenvectors : bool
            If False, return only eigenvalues.

        Returns
        -------
        evals (and evecs if requested)
        """
        # Prefer Hermitian solver if flagged as Hermitian
        if self.is_hermitian:
            sp_linalg = _require_scipy_linalg("ground_state")
            # Use sparse if possible for scalability
            try:
                sp = self.to_sparse()
                evals, evecs = sp_linalg.eigsh(sp, k=k, which=which, maxiter=maxiter, tol=tol)
            except NotImplementedError:
                # wrap matvec in LinearOperator
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

    # ---------- Niceties ----------
    def __repr__(self) -> str:
        cls = self.__class__.__name__
        try:
            p = dict(self.parameters())
        except Exception:
            p = {}
        summary = ", ".join(f"{k}={v}" for k, v in p.items()) if p else "…"
        return f"{cls}(N={self.num_sites}, dim={self.hilbert_dim}, dtype={self.dtype}, params={{ {summary} }})"


# -------------------- internal helpers --------------------

def _require_scipy_linalg(where: str):
    """
    Import scipy.sparse.linalg lazily with a clear error if missing.
    """
    try:
        import scipy.sparse.linalg as sp_linalg  # type: ignore
        return sp_linalg
    except Exception as e:
        raise RuntimeError(
            f"{where} requires SciPy. Please install with `pip install scipy`."
        ) from e