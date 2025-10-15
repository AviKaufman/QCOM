# tests/test_metrics_entanglement.py
import numpy as np
import pytest

from qcom.metrics.entanglement import (
    von_neumann_entropy_from_rdm,
    von_neumann_entropy_from_state,
    von_neumann_entropy_from_hamiltonian,
)


# -------------------- Helpers --------------------

def bell_phi_plus():
    """Return |Φ+> = (|00> + |11>)/sqrt(2) as a 4-vector."""
    psi = np.zeros(4, dtype=np.complex128)
    psi[0] = 1.0 / np.sqrt(2.0)  # |00>
    psi[3] = 1.0 / np.sqrt(2.0)  # |11>
    return psi


def projector(psi: np.ndarray) -> np.ndarray:
    """Return |psi><psi| (normalized if needed)."""
    psi = np.asarray(psi, dtype=np.complex128).reshape(-1)
    norm = np.linalg.norm(psi)
    if norm == 0:
        raise ValueError("Zero vector cannot form a projector.")
    psi = psi / norm
    return np.outer(psi, np.conjugate(psi))


def partial_trace_one_qubit(rho: np.ndarray, keep: int) -> np.ndarray:
    """
    Partial trace of a 2-qubit density matrix, keeping qubit `keep` in {0,1}.
    Convention: basis ordering |00>, |01>, |10>, |11>, with qubit-0 as MSB.
    """
    rho = rho.reshape(2, 2, 2, 2)  # (i0, i1, j0, j1)
    if keep == 0:
        # trace out qubit 1 → sum over i1 == j1
        return np.einsum("i0 i1 j0 j1, i1 j1 -> i0 j0", rho, np.eye(2))
    elif keep == 1:
        # trace out qubit 0 → sum over i0 == j0
        return np.einsum("i0 i1 j0 j1, i0 j0 -> i1 j1", rho, np.eye(2))
    else:
        raise ValueError("keep must be 0 or 1")


def _isclose(a, b, rtol=1e-10, atol=1e-12):
    return np.isclose(a, b, rtol=rtol, atol=atol)


# -------------------- von_neumann_entropy_from_rdm --------------------

def test_entropy_rdm_handles_zero_eigenvalues_safely():
    # RDM with eigenvalues [1, 0] ⇒ entropy 0 (no log(0) issues)
    rdm = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=float)
    S = von_neumann_entropy_from_rdm(rdm, base=2)
    assert _isclose(S, 0.0)


# -------------------- von_neumann_entropy_from_state --------------------

def test_entropy_from_state_product_zero():
    psi = np.array([1, 0, 0, 0], dtype=np.complex128)  # |00>
    config_keep_qubit0 = [1, 0]  # keep qubit 0, trace qubit 1
    S = von_neumann_entropy_from_state(psi, configuration=config_keep_qubit0, base=2)
    assert _isclose(S, 0.0)

def test_entropy_from_state_bell_one_bit():
    psi = bell_phi_plus()
    # keep qubit 0 (1), trace qubit 1 (0)
    config_keep_qubit0 = [1, 0]
    S = von_neumann_entropy_from_state(psi, configuration=config_keep_qubit0, base=2)
    assert _isclose(S, 1.0)

def test_entropy_from_state_validates_power_of_two_and_config_len():
    with pytest.raises(ValueError, match="not a power of 2"):
        von_neumann_entropy_from_state(np.array([1, 0, 0], dtype=np.complex128), configuration=[1, 0])
    with pytest.raises(ValueError, match="must equal the number of qubits"):
        von_neumann_entropy_from_state(np.array([1, 0, 0, 0], dtype=np.complex128), configuration=[1, 0, 0])


# -------------------- von_neumann_entropy_from_hamiltonian --------------------

def test_entropy_from_hamiltonian_bell_ground_state():
    # H = -|Φ+><Φ+| → unique ground state is |Φ+>
    psi = bell_phi_plus()
    H = -projector(psi)  # 4x4 dense ndarray, accepted by as_linear_operator
    config_keep_qubit0 = [1, 0]  # keep qubit 0
    S = von_neumann_entropy_from_hamiltonian(
        H, configuration=config_keep_qubit0, state_index=0, base=2, show_progress=False
    )
    assert _isclose(S, 1.0)

def test_entropy_from_hamiltonian_product_ground_state_zero():
    # H = -|00><00| → ground state is |00>, product ⇒ S = 0
    psi00 = np.array([1, 0, 0, 0], dtype=np.complex128)
    H = -projector(psi00)
    config_keep_qubit0 = [1, 0]
    S = von_neumann_entropy_from_hamiltonian(
        H, configuration=config_keep_qubit0, state_index=0, base=2, show_progress=False
    )
    assert _isclose(S, 0.0)

def test_entropy_from_hamiltonian_validates_dim_and_config():
    psi00 = np.array([1, 0, 0, 0], dtype=np.complex128)
    H = -projector(psi00)
    with pytest.raises(ValueError, match="must equal the number of qubits"):
        von_neumann_entropy_from_hamiltonian(H, configuration=[1, 0, 0], base=2)