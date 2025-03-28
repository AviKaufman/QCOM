import numpy as np
import pytest
from scipy.sparse import csr_matrix
import qcom as qc

"""
This file is for testing the Hamiltonian functions in qcom/hamiltonians.py using pytest.
To run the tests, use the command `pytest tests/test_hamiltonians.py` in the root directory of the repository.
If you get an error, do not push the changes to GitHub until the error is fixed.
"""

# --- Tests for build_rydberg_hamiltonian_chain --- #


def test_build_rydberg_hamiltonian_chain_no_progress():
    # Use a small chain of 2 atoms.
    num_atoms = 2
    Omega = 1.0
    Delta = 1.0
    a = 1.0
    pbc = False
    H = qc.build_rydberg_hamiltonian_chain(
        num_atoms, Omega, Delta, a, pbc, show_progress=False
    )
    # Check type and shape
    assert isinstance(H, csr_matrix)
    expected_dim = 2**num_atoms
    assert H.shape == (expected_dim, expected_dim)
    # Check Hermiticity: dense version equals its transpose.
    H_dense = H.todense()
    assert np.allclose(H_dense, H_dense.T, atol=1e-6)


def test_build_rydberg_hamiltonian_chain_with_progress(capsys):
    # Same as above but with progress enabled.
    num_atoms = 2
    Omega = 1.0
    Delta = 1.0
    a = 1.0
    pbc = False
    H = qc.build_rydberg_hamiltonian_chain(
        num_atoms, Omega, Delta, a, pbc, show_progress=True
    )
    captured = capsys.readouterr().out
    assert "Building Rydberg Hamiltonian (Chain)" in captured
    # Check shape and Hermiticity.
    expected_dim = 2**num_atoms
    assert H.shape == (expected_dim, expected_dim)
    H_dense = H.todense()
    assert np.allclose(H_dense, H_dense.T, atol=1e-6)


# --- Tests for build_rydberg_hamiltonian_ladder --- #


def test_build_rydberg_hamiltonian_ladder_no_progress():
    # For ladder configuration, use an even number of atoms (e.g. 4).
    num_atoms = 4
    Omega = 1.0
    Delta = 1.0
    a = 1.0
    rho = 2.0
    pbc = False
    H = qc.build_rydberg_hamiltonian_ladder(
        num_atoms, Omega, Delta, a, rho, pbc, show_progress=False
    )
    assert isinstance(H, csr_matrix)
    expected_dim = 2**num_atoms
    assert H.shape == (expected_dim, expected_dim)
    H_dense = H.todense()
    assert np.allclose(H_dense, H_dense.T, atol=1e-6)


def test_build_rydberg_hamiltonian_ladder_invalid_num_atoms():
    # For a ladder, the number of atoms must be even.
    with pytest.raises(AssertionError):
        qc.build_rydberg_hamiltonian_ladder(
            3, 1.0, 1.0, 1.0, rho=2, pbc=False, show_progress=False
        )


# --- Tests for build_ising_hamiltonian --- #


def test_build_ising_hamiltonian_no_progress():
    # Test a simple Ising chain with 2 spins.
    num_spins = 2
    J = 1.0
    h_field = 1.0
    pbc = False
    H = qc.build_ising_hamiltonian(num_spins, J, h_field, pbc, show_progress=False)
    assert isinstance(H, csr_matrix)
    expected_dim = 2**num_spins
    assert H.shape == (expected_dim, expected_dim)
    H_dense = H.todense()
    # Ising Hamiltonian should be symmetric.
    assert np.allclose(H_dense, H_dense.T, atol=1e-6)


def test_build_ising_hamiltonian_with_progress(capsys):
    num_spins = 3
    J = 1.0
    h_field = 1.0
    pbc = False
    H = qc.build_ising_hamiltonian(num_spins, J, h_field, pbc, show_progress=True)
    captured = capsys.readouterr().out
    assert "Building Ising Hamiltonian (1D Chain)" in captured
    expected_dim = 2**num_spins
    assert H.shape == (expected_dim, expected_dim)
    H_dense = H.todense()
    assert np.allclose(H_dense, H_dense.T, atol=1e-6)


# --- Tests for build_ising_hamiltonian_ladder --- #


def test_build_ising_hamiltonian_ladder_no_progress():
    # For a ladder, use an even number of spins, e.g., 4.
    num_spins = 4
    J = 1.0
    h_field = 1.0
    pbc = False
    include_diagonal = True
    H = qc.build_ising_hamiltonian_ladder(
        num_spins, J, h_field, pbc, include_diagonal, show_progress=False
    )
    assert isinstance(H, csr_matrix)
    expected_dim = 2**num_spins
    assert H.shape == (expected_dim, expected_dim)
    H_dense = H.todense()
    assert np.allclose(H_dense, H_dense.T, atol=1e-6)


def test_build_ising_hamiltonian_ladder_invalid_num_spins():
    # Ladder requires an even number of spins.
    with pytest.raises(AssertionError):
        qc.build_ising_hamiltonian_ladder(
            3, 1.0, 1.0, pbc=False, include_diagonal=True, show_progress=False
        )


def test_build_ising_hamiltonian_ladder_with_progress(capsys):
    num_spins = 4
    J = 1.0
    h_field = 1.0
    pbc = False
    include_diagonal = True
    H = qc.build_ising_hamiltonian_ladder(
        num_spins, J, h_field, pbc, include_diagonal, show_progress=True
    )
    captured = capsys.readouterr().out
    assert "Building Ising Hamiltonian (Ladder)" in captured
    expected_dim = 2**num_spins
    assert H.shape == (expected_dim, expected_dim)
    H_dense = H.todense()
    assert np.allclose(H_dense, H_dense.T, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__])
