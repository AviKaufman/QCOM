# tests/test_solvers_static.py
import numpy as np
import pytest

scipy = pytest.importorskip("scipy")  # static solvers rely on SciPy
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from qcom.solvers.static import (
    as_linear_operator,
    eigensolve,
    ground_state,
    find_eigenstate,
    full_dense_spectrum,
)


# -------------------- as_linear_operator --------------------

def test_as_linear_operator_dense_matvec_and_dim():
    A = np.array([[2.0, -1.0], [-1.0, 2.0]], dtype=float)
    Lop, is_herm, dim = as_linear_operator(A)
    assert dim == 2
    assert Lop.shape == (2, 2)
    # Sanity: matvec matches A @ x
    x = np.array([1.0, 3.0])
    Ax = A @ x
    assert np.allclose(Lop @ x, Ax)
    # Hint may be None for dense; allowed
    assert is_herm in (None, True, False)

def test_as_linear_operator_sparse_matvec_and_dim():
    A = sp.csr_matrix(np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float))
    Lop, is_herm, dim = as_linear_operator(A)
    assert dim == 2 and Lop.shape == (2, 2)
    x = np.array([1.0, 2.0])
    assert np.allclose(Lop @ x, (A @ x))
    assert is_herm in (None, True, False)


# -------------------- eigensolve core --------------------

def test_eigensolve_hermitian_matches_numpy_eigh_smallest_two():
    rng = np.random.default_rng(0)
    M = rng.normal(size=(5, 5))
    H = (M + M.T) / 2.0  # Hermitian (real symmetric)
    # NumPy ground truth
    w_full, v_full = np.linalg.eigh(H)
    w_np_small2 = w_full[:2]

    # Our solver (Hermitian path)
    evals, evecs = eigensolve(H, k=2, which="SA", hermitian=True, return_eigenvectors=True)
    w = np.sort(np.real(evals))
    assert np.allclose(w, w_np_small2, atol=1e-8)

    # Check eigenvector spans (up to unitary mixing): verify Av ≈ λv for each
    for i in range(2):
        lam = np.real(evals[i])
        vec = evecs[:, i]
        assert np.allclose(H @ vec, lam * vec, atol=1e-6)


# -------------------- ground_state & find_eigenstate --------------------

def test_ground_state_on_diagonal():
    # Diagonal Hamiltonian with obvious spectrum
    d = np.array([0.0, 2.0, 5.0, 9.0], dtype=float)
    H = np.diag(d)
    e0, v0 = ground_state(H, return_vector=True)
    assert np.isclose(e0, 0.0)
    # v0 should be close to |000...> basis vector e_0 (up to a phase)
    e0_vec = np.zeros_like(d, dtype=np.complex128); e0_vec[0] = 1.0
    # Compare up to a global phase: use absolute inner product
    overlap = np.abs(np.vdot(e0_vec, v0))
    assert np.isclose(overlap, 1.0, atol=1e-8)

def test_find_eigenstate_by_index():
    d = np.array([0.0, 0.5, 2.0, 7.0], dtype=float)
    H = np.diag(d)
    e1, v1 = find_eigenstate(H, state_index=1)
    assert np.isclose(e1, 0.5, atol=1e-12)
    e1_vec = np.zeros_like(d, dtype=np.complex128); e1_vec[1] = 1.0
    assert np.isclose(np.abs(np.vdot(e1_vec, v1)), 1.0, atol=1e-8)


# -------------------- full_dense_spectrum --------------------

def test_full_dense_spectrum_matches_numpy_hermitian():
    rng = np.random.default_rng(1)
    M = rng.normal(size=(4, 4))
    H = (M + M.T) / 2.0
    w_np, V_np = np.linalg.eigh(H)

    w, V = full_dense_spectrum(H, return_eigenvectors=True)
    assert np.allclose(w, w_np, atol=1e-12)
    # Eigenvectors may differ by column phases; check Av ≈ λv spot-check
    for i in range(4):
        assert np.allclose(H @ V[:, i], w[i] * V[:, i], atol=1e-10)

def test_full_dense_spectrum_sparse_input_and_threshold_error():
    # Use a modest size to avoid heavy allocations but still trigger max_dense_dim
    n = 33
    Hs = sp.eye(n, format="csr")
    # With max_dense_dim < n → should raise before/when validating dimension
    with pytest.raises(ValueError):
        full_dense_spectrum(Hs, max_dense_dim=32, return_eigenvectors=False)
    # With sufficient threshold it should succeed and return all ones
    w = full_dense_spectrum(Hs, max_dense_dim=128, return_eigenvectors=False)
    assert w.shape == (n,)
    assert np.allclose(np.sort(w), np.ones(n))

def test_full_dense_spectrum_from_linear_operator():
    # Build a LinearOperator representing a known dense matrix
    A = np.array([[0.0, 2.0], [2.0, 3.0]], dtype=np.complex128)
    Lop = spla.LinearOperator(shape=A.shape, matvec=(lambda x: A @ x), dtype=A.dtype)

    w, V = full_dense_spectrum(Lop, max_dense_dim=16, return_eigenvectors=True)
    w_np, V_np = np.linalg.eigh(A)
    assert np.allclose(w, w_np, atol=1e-12)
    for i in range(2):
        assert np.allclose(A @ V[:, i], w[i] * V[:, i], atol=1e-12)