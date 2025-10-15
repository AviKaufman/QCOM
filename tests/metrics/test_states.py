# tests/test_metrics_states.py
import numpy as np
import pytest

from qcom.metrics.states import create_density_matrix, compute_reduced_density_matrix


# -------------------- Helpers --------------------

def is_hermitian(M, tol=1e-12):
    return np.allclose(M, M.conj().T, atol=tol, rtol=0)

def is_psd(M, tol=1e-12):
    # Numerical PSD check: all eigenvalues >= -tol
    evals = np.linalg.eigvalsh(M)
    return np.all(evals >= -tol)


# -------------------- create_density_matrix --------------------

def test_create_density_matrix_basic_properties():
    # |psi> = (|00> + i|11>) / sqrt(2)
    psi = np.array([1.0, 0.0, 0.0, 1.0j], dtype=np.complex128) / np.sqrt(2)
    rho = create_density_matrix(psi, show_progress=True)

    # shape
    assert rho.shape == (4, 4)

    # Hermitian
    assert is_hermitian(rho)

    # trace = 1
    assert np.isclose(np.trace(rho), 1.0)

    # rank-1 projector: rho^2 == rho (within numeric tol)
    assert np.allclose(rho @ rho, rho, atol=1e-12, rtol=0)


def test_create_density_matrix_real_vector_yields_real_matrix():
    # |psi> = |01>
    psi = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
    rho = create_density_matrix(psi)
    # Exactly one element on diagonal = 1
    assert np.isclose(np.trace(rho), 1.0)
    assert np.isrealobj(rho)


# -------------------- compute_reduced_density_matrix --------------------

def test_partial_trace_two_qubits_trace_out_one():
    # |psi> = (|00> + |11>) / sqrt(2) → Bell state
    psi = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.complex128) / np.sqrt(2)
    rho = create_density_matrix(psi)

    # configuration: keep left (site 0), trace out right (site 1)
    # Convention: configuration[i] == 1 → keep
    # For two qubits, e.g., [1, 0] keeps MSB / left qubit
    rdm = compute_reduced_density_matrix(rho, configuration=[1, 0], show_progress=True)

    # Result should be maximally mixed 2x2: I/2
    assert rdm.shape == (2, 2)
    assert is_hermitian(rdm)
    assert is_psd(rdm)
    assert np.allclose(rdm, 0.5 * np.eye(2))

def test_partial_trace_two_qubits_keep_both_is_identity_mapping():
    # |psi> = |01>
    psi = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.complex128)
    rho = create_density_matrix(psi)
    rdm = compute_reduced_density_matrix(rho, configuration=[1, 1])
    # Keeping both sites → RDM == rho
    assert rdm.shape == (4, 4)
    assert np.allclose(rdm, rho)

def test_partial_trace_two_qubits_trace_both_returns_scalar_one_by_one():
    # Any normalized pure state should reduce to [[1.0]] if all sites traced out
    psi = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.complex128)
    rho = create_density_matrix(psi)
    rdm = compute_reduced_density_matrix(rho, configuration=[0, 0])
    assert rdm.shape == (1, 1)
    # Trace preserved
    assert np.isclose(rdm[0, 0], 1.0)

def test_partial_trace_three_qubits_keep_middle_only():
    # N=3, index order: MSB ↔ site 0, then site 1, then LSB ↔ site 2
    # Build |psi> = |010> (basis index 0b010 = 2)
    psi = np.zeros(8, dtype=np.complex128)
    psi[2] = 1.0  # |010>
    rho = create_density_matrix(psi)

    # Keep only site 1 (middle): configuration [0,1,0]
    rdm = compute_reduced_density_matrix(rho, configuration=[0, 1, 0])

    # The kept qubit is |1> deterministically → RDM = |1><1|
    assert rdm.shape == (2, 2)
    expected = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
    assert np.allclose(rdm, expected)
    assert is_psd(rdm)

def test_partial_trace_preserves_trace_for_mixed_state():
    # Build a mixed rho on 2 qubits: 0.3 |00><00| + 0.7 |11><11|
    rho = np.zeros((4, 4), dtype=np.complex128)
    rho[0, 0] = 0.3
    rho[3, 3] = 0.7
    # Keep left qubit only: [1,0]
    rdm = compute_reduced_density_matrix(rho, configuration=[1, 0])
    assert np.isclose(np.trace(rdm), 1.0)
    assert is_psd(rdm)
    assert is_hermitian(rdm)
    # The marginal should be diag([0.3, 0.7])
    expected = np.diag([0.3, 0.7])
    assert np.allclose(rdm, expected)

# -------------------- Error handling --------------------

def test_compute_reduced_density_matrix_raises_on_bad_shape():
    # Not a power-of-two dimension (3x3)
    bad = np.eye(3)
    with pytest.raises(ValueError):
        compute_reduced_density_matrix(bad, configuration=[1, 1])

def test_compute_reduced_density_matrix_raises_on_wrong_config_length():
    psi = np.array([1.0, 0.0], dtype=np.complex128)  # 1 qubit
    rho = create_density_matrix(psi)
    with pytest.raises(ValueError):
        compute_reduced_density_matrix(rho, configuration=[1, 1])  # wrong length