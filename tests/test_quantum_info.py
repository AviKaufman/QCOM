import numpy as np
import pytest
import qcom as qc

"""
This file is for testing the quantum information functions in qcom/quantum_info.py using pytest.
To run the tests, use the command `pytest tests/test_quantum_info.py` in the root directory of the repository.
If you get an error, do not push the changes to GitHub until the error is fixed.
"""

# -------------------------------
# von_neumann_entropy_from_rdm
# -------------------------------


def test_von_neumann_entropy_from_rdm_pure():
    # Pure state: density matrix with eigenvalues [1, 0] should have entropy 0.
    rdm = np.array([[1, 0], [0, 0]], dtype=float)
    entropy = qc.von_neumann_entropy_from_rdm(rdm)
    assert np.isclose(entropy, 0.0, atol=1e-6)


def test_von_neumann_entropy_from_rdm_mixed():
    # Maximally mixed state for a qubit: eigenvalues 0.5, 0.5, so entropy = -2*0.5*ln(0.5) = ln(2).
    rdm = np.array([[0.5, 0], [0, 0.5]], dtype=float)
    expected = -(0.5 * np.log(0.5) + 0.5 * np.log(0.5))
    entropy = qc.von_neumann_entropy_from_rdm(rdm)
    assert np.isclose(entropy, expected, atol=1e-6)


# ----------------------------------------------------
# von_neumann_entropy_from_hamiltonian
# ----------------------------------------------------


def test_von_neumann_entropy_from_hamiltonian_no_progress():
    # Use a simple 1-qubit Hamiltonian.
    # For H = [[0,0],[0,1]], the lowest eigenstate is [1,0],
    # so the density matrix is |0><0|, a pure state with entropy 0.
    H = np.array([[0, 0], [0, 1]], dtype=float)
    configuration = [1]  # Keep the only qubit.
    entropy = qc.von_neumann_entropy_from_hamiltonian(
        H, configuration, state_index=0, show_progress=False
    )
    assert np.isclose(entropy, 0.0, atol=1e-6)


def test_von_neumann_entropy_from_hamiltonian_with_progress(capsys):
    H = np.array([[0, 0], [0, 1]], dtype=float)
    configuration = [1]
    entropy = qc.von_neumann_entropy_from_hamiltonian(
        H, configuration, state_index=0, show_progress=True
    )
    captured = capsys.readouterr().out
    assert "Computing Von Neumann Entropy" in captured
    assert np.isclose(entropy, 0.0, atol=1e-6)


# ----------------------------------------------------
# get_eigenstate_probabilities
# ----------------------------------------------------


def test_get_eigenstate_probabilities_no_progress():
    # For the same 1-qubit Hamiltonian, lowest eigenstate is [1, 0].
    H = np.array([[0, 0], [0, 1]], dtype=float)
    prob_dict = qc.get_eigenstate_probabilities(H, state_index=0, show_progress=False)
    expected = {"0": 1.0, "1": 0.0}
    for key in expected:
        assert np.isclose(prob_dict[key], expected[key], atol=1e-6)


def test_get_eigenstate_probabilities_with_progress(capsys):
    H = np.array([[0, 0], [0, 1]], dtype=float)
    prob_dict = qc.get_eigenstate_probabilities(H, state_index=0, show_progress=True)
    captured = capsys.readouterr().out
    assert "Computing Eigenstate Probabilities" in captured
    expected = {"0": 1.0, "1": 0.0}
    for key in expected:
        assert np.isclose(prob_dict[key], expected[key], atol=1e-6)


# ----------------------------------------------------
# create_density_matrix
# ----------------------------------------------------


def test_create_density_matrix_no_progress():
    eigenvector = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=complex)
    density_matrix = qc.create_density_matrix(eigenvector, show_progress=False)
    expected = np.outer(eigenvector, np.conjugate(eigenvector))
    assert np.allclose(density_matrix, expected, atol=1e-6)


def test_create_density_matrix_with_progress(capsys):
    eigenvector = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=complex)
    density_matrix = qc.create_density_matrix(eigenvector, show_progress=True)
    captured = capsys.readouterr().out
    assert "Constructing Density Matrix" in captured
    expected = np.outer(eigenvector, np.conjugate(eigenvector))
    assert np.allclose(density_matrix, expected, atol=1e-6)


# ----------------------------------------------------
# compute_reduced_density_matrix
# ----------------------------------------------------


def test_compute_reduced_density_matrix_one_qubit_no_trace():
    # For a 1-qubit system, configuration [1] means keep the qubit.
    density_matrix = np.array([[1, 0], [0, 0]], dtype=float)
    reduced = qc.compute_reduced_density_matrix(
        density_matrix, configuration=[1], show_progress=False
    )
    # Should remain unchanged.
    assert np.allclose(reduced, density_matrix, atol=1e-6)


def test_compute_reduced_density_matrix_one_qubit_trace():
    # For a 1-qubit system, configuration [0] traces out the qubit, yielding a 1x1 matrix [1].
    density_matrix = np.array([[1, 0], [0, 0]], dtype=float)
    reduced = qc.compute_reduced_density_matrix(
        density_matrix, configuration=[0], show_progress=False
    )
    assert reduced.shape == (1, 1)
    assert np.allclose(reduced, np.array([[1]], dtype=float), atol=1e-6)


def test_compute_reduced_density_matrix_two_qubits():
    # For a 2-qubit pure state |00> = [1, 0, 0, 0]
    eigenvector = np.zeros(4)
    eigenvector[0] = 1.0
    density_matrix = np.outer(eigenvector, eigenvector)
    # Use configuration [0, 1] to trace out the first qubit (configuration[0]==0) and keep the second.
    reduced = qc.compute_reduced_density_matrix(
        density_matrix, configuration=[0, 1], show_progress=False
    )
    # The reduced density matrix for qubit (from state |00>) should be |0><0| = [[1,0],[0,0]].
    expected = np.array([[1, 0], [0, 0]], dtype=float)
    assert reduced.shape == (2, 2)
    assert np.allclose(reduced, expected, atol=1e-6)


def test_compute_reduced_density_matrix_with_progress(capsys):
    density_matrix = np.array([[1, 0], [0, 0]], dtype=float)
    reduced = qc.compute_reduced_density_matrix(
        density_matrix, configuration=[1], show_progress=True
    )
    captured = capsys.readouterr().out
    assert "Computing Reduced Density Matrix" in captured
    assert np.allclose(reduced, density_matrix, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__])
