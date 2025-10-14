# qcom/solvers/static.py
"""
Static (time-independent) eigen solvers for QCOM.

Purpose
-------
Provide thin-spectrum and convenience routines to extract eigenvalues
and eigenvectors of time-independent Hamiltonians. Inputs can be:
- NumPy dense arrays
- SciPy sparse matrices
- qcom.hamiltonians.BaseHamiltonian instances (materialized lazily)

Main API
--------
- as_linear_operator(H)        -> (LinearOperator, is_hermitian: bool|None, dim: int)
- eigensolve(H, k=1, which="SA", sigma=None, hermitian=None, tol=0.0, maxiter=None, return_eigenvectors=True, show_progress=False)
- ground_state(H, return_vector=True, **kwargs) -> (e0, v0) or e0
- find_eigenstate(H, state_index=0, show_progress=False)  # compat helper
- full_dense_spectrum(H, max_dense_dim=4096) -> (evals, evecs)

Notes
-----
- Uses SciPy’s sparse eigensolvers (`eigsh` for Hermitian, `eigs` for general).
- For small problems you can call `full_dense_spectrum` to get the entire spectrum.
- We keep progress printing light and nest-safe via ProgressManager.
"""

from __future__ import annotations
import time
import numpy as np
from typing import Optional, Tuple

from .._internal.progress import ProgressManager

# -------------------- Optional SciPy imports (lazy, with clear error) --------------------

def _require_scipy_linalg(where: str):
    try:
        import scipy.sparse.linalg as sp_linalg  # type: ignore
        return sp_linalg
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"{where} requires SciPy. Install with `pip install scipy`.") from e

def _maybe_is_sparse_matrix(H) -> bool:
    try:
        import scipy.sparse as sp  # type: ignore
    except Exception:
        return False
    return sp.issparse(H)

# -------------------- Normalization: get LinearOperator from various inputs --------------------

def as_linear_operator(H) -> Tuple["object", Optional[bool], int]:
    """
    Normalize a Hamiltonian-like input into a SciPy LinearOperator.

    Args:
        H: np.ndarray, SciPy sparse matrix, or BaseHamiltonian.

    Returns:
        (LinearOperator, is_hermitian, dim)
        - is_hermitian may be None if unknown.

    Raises:
        RuntimeError if SciPy not available.
        ValueError for incompatible shapes.
    """
    sp_linalg = _require_scipy_linalg("as_linear_operator")

    # qcom BaseHamiltonian?
    try:
        from ..hamiltonians.base import BaseHamiltonian  # local import, no cycle at top-level
        if isinstance(H, BaseHamiltonian):
            Lop = H.to_linear_operator()
            try:
                is_herm = bool(H.is_hermitian)
            except Exception:
                is_herm = None
            return Lop, is_herm, H.hilbert_dim
    except Exception:
        pass  # fall through

    # NumPy dense?
    if isinstance(H, np.ndarray):
        if H.ndim != 2 or H.shape[0] != H.shape[1]:
            raise ValueError("Dense Hamiltonian must be a square 2D array.")
        dim = H.shape[0]
        dtype = H.dtype

        def mv(x):
            return H @ x

        Lop = sp_linalg.LinearOperator(shape=(dim, dim), matvec=mv, dtype=dtype)
        # Try a cheap Hermitian check when real/complex is simple (optional)
        is_herm = None
        if np.isrealobj(H):
            # Heuristic to avoid O(d^2): check symmetry of a few rows/cols
            # (Leave as None to avoid surprises; users can override.)
            is_herm = None
        return Lop, is_herm, dim

    # SciPy sparse?
    if _maybe_is_sparse_matrix(H):
        dim = H.shape[0]
        dtype = H.dtype

        def mv(x):
            return H @ x

        Lop = sp_linalg.LinearOperator(shape=(dim, dim), matvec=mv, dtype=dtype)
        return Lop, None, dim

    # LinearOperator already?
    try:
        import scipy.sparse.linalg as sp_linalg  # type: ignore
        if isinstance(H, sp_linalg.LinearOperator):
            if H.shape[0] != H.shape[1]:
                raise ValueError("LinearOperator must be square.")
            return H, None, H.shape[0]
    except Exception:
        pass

    raise TypeError(
        "Unsupported Hamiltonian type. Provide a NumPy array, SciPy sparse matrix, "
        "LinearOperator, or a qcom BaseHamiltonian."
    )

# -------------------- Core thin-spectrum solver --------------------

def eigensolve(
    H,
    *,
    k: int = 1,
    which: str = "SA",
    sigma: Optional[float] = None,
    hermitian: Optional[bool] = None,
    tol: float = 0.0,
    maxiter: Optional[int] = None,
    return_eigenvectors: bool = True,
    show_progress: bool = False,
):
    """
    Compute k extremal eigenvalues/eigenvectors.

    Args:
        H: Hamiltonian-like (np.ndarray, SciPy sparse, LinearOperator, or BaseHamiltonian).
        k: number of eigenpairs.
        which: eigensolver selector ("SA","LA","SM","LM", etc.). For Hermitian problems,
               `eigsh` uses "SA", "LA", "BE"…; for non-Hermitian, `eigs` semantics apply.
        sigma: shift-invert target; enables interior eigenpairs near `sigma`.
        hermitian: override Hermitian flag. If None, we use the input’s hint when available.
        tol: solver tolerance.
        maxiter: solver iteration cap.
        return_eigenvectors: if False, return only eigenvalues.
        show_progress: show a simple progress scope around the call.

    Returns:
        evals (and evecs if requested)
    """
    sp_linalg = _require_scipy_linalg("eigensolve")
    Lop, hint_herm, dim = as_linear_operator(H)
    use_herm = hermitian if hermitian is not None else bool(hint_herm)

    with (ProgressManager.progress("Eigen solve", total_steps=1) if show_progress else ProgressManager.dummy_context()):
        t0 = time.time()

        if use_herm:
            # Hermitian solver
            evals, evecs = sp_linalg.eigsh(
                Lop,
                k=k,
                which=which,
                sigma=sigma,
                tol=tol,
                maxiter=maxiter,
                return_eigenvectors=True,
            )
        else:
            # General (possibly non-Hermitian) solver
            evals, evecs = sp_linalg.eigs(
                Lop,
                k=k,
                which=which,
                sigma=sigma,
                tol=tol,
                maxiter=maxiter,
                return_eigenvectors=True,
            )

        if show_progress:
            _ = time.time() - t0
            ProgressManager.update_progress(1)

    if return_eigenvectors:
        return evals, evecs
    return evals

# -------------------- Convenience: ground state --------------------

def ground_state(
    H,
    *,
    which: str = "SA",
    return_vector: bool = True,
    tol: float = 0.0,
    maxiter: Optional[int] = None,
    show_progress: bool = False,
):
    """
    Return the lowest eigenvalue (and eigenvector if requested).

    Notes:
        Uses Hermitian solver by default; override by calling eigensolve(..., hermitian=False)
        if your H is intentionally non-Hermitian.
    """
    evals, evecs = eigensolve(
        H,
        k=1,
        which=which,
        hermitian=True,   # ground state implies Hermitian in our typical workflows
        tol=tol,
        maxiter=maxiter,
        return_eigenvectors=True,
        show_progress=show_progress,
    )
    if return_vector:
        return float(np.real(evals[0])), evecs[:, 0]
    return float(np.real(evals[0]))

# -------------------- Back-compat helper: eigenstate by index --------------------

def find_eigenstate(hamiltonian, state_index: int = 0, show_progress: bool = False):
    """
    Return (eigenvalue, eigenvector) for the eigenstate at `state_index`
    when eigenvalues are sorted in ascending order.

    Implementation detail:
        Uses `eigsh` in Hermitian mode and asks for k=state_index+1, then selects the
        final column. This matches the behavior of the previous utils.find_eigenstate.
    """
    if state_index < 0:
        raise ValueError("state_index must be non-negative.")

    # Request the first (state_index+1) eigenpairs and pick the last
    evals, evecs = eigensolve(
        hamiltonian,
        k=state_index + 1,
        which="SA",
        hermitian=True,
        return_eigenvectors=True,
        show_progress=show_progress,
    )
    return float(np.real(evals[state_index])), evecs[:, state_index]

# -------------------- Full spectrum for small problems --------------------

def full_dense_spectrum(H, *, max_dense_dim: int = 4096, return_eigenvectors: bool = True):
    """
    Compute the entire spectrum (and optionally eigenvectors) via dense linear algebra.

    Args:
        H: np.ndarray, SciPy sparse matrix (will be densified), or BaseHamiltonian.
        max_dense_dim: safety limit; if dimension exceeds this, raise an error.
        return_eigenvectors: whether to return eigenvectors.

    Returns:
        (evals, evecs) or (evals,) if return_eigenvectors=False

    Raises:
        ValueError if dimension exceeds `max_dense_dim`.
    """
    # Materialize to dense
    if isinstance(H, np.ndarray):
        A = H
    elif _maybe_is_sparse_matrix(H):
        A = H.toarray()
    else:
        # BaseHamiltonian or LinearOperator → try to_dense via BaseHamiltonian if available
        try:
            from ..hamiltonians.base import BaseHamiltonian
            if isinstance(H, BaseHamiltonian):
                A = H.to_dense()
            else:
                # Last resort: form dense by applying LinearOperator to identity (costly).
                Lop, _, dim = as_linear_operator(H)
                if dim > max_dense_dim:
                    raise ValueError(
                        f"full_dense_spectrum: dimension {dim} exceeds max_dense_dim={max_dense_dim}"
                    )
                A = np.empty((dim, dim), dtype=np.complex128)
                I = np.eye(dim, dtype=np.complex128)
                for j in range(dim):
                    A[:, j] = Lop @ I[:, j]
        except Exception:
            # Try generic path via as_linear_operator
            Lop, _, dim = as_linear_operator(H)
            if dim > max_dense_dim:
                raise ValueError(
                    f"full_dense_spectrum: dimension {dim} exceeds max_dense_dim={max_dense_dim}"
                )
            A = np.empty((dim, dim), dtype=np.complex128)
            I = np.eye(dim, dtype=np.complex128)
            for j in range(dim):
                A[:, j] = Lop @ I[:, j]

    if A.shape[0] != A.shape[1]:
        raise ValueError("Dense Hamiltonian must be square.")

    dim = A.shape[0]
    if dim > max_dense_dim:
        raise ValueError(
            f"full_dense_spectrum: dimension {dim} exceeds max_dense_dim={max_dense_dim}"
        )

    # Hermitian path if numerically Hermitian (cheap check)
    if np.allclose(A, A.conj().T):
        if return_eigenvectors:
            evals, evecs = np.linalg.eigh(A)
            return evals, evecs
        return np.linalg.eigvalsh(A)

    # General case
    if return_eigenvectors:
        evals, evecs = np.linalg.eig(A)
        return evals, evecs
    return np.linalg.eigvals(A)