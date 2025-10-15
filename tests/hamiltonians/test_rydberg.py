# tests/test_rydberg_builder.py
import numpy as np
import pytest

from qcom.hamiltonians.rydberg import (
    build_rydberg,
    RydbergHamiltonian,
)


# --------------------------------------------------------------------------------------
# Minimal register used for tests (duck-typed: len(), .sites, .distances()).
# --------------------------------------------------------------------------------------
class FakeRegister:
    def __init__(self, coords: np.ndarray):
        """
        coords: shape (N, d), positions in meters
        """
        self.sites = np.asarray(coords, dtype=float)

    def __len__(self) -> int:
        return 0 if self.sites.size == 0 else self.sites.shape[0]

    def distances(self) -> np.ndarray:
        if len(self) == 0:
            return np.zeros((0, 0), dtype=float)
        X = self.sites
        # pairwise Euclidean distances
        diffs = X[:, None, :] - X[None, :, :]
        D = np.sqrt(np.sum(diffs**2, axis=-1))
        np.fill_diagonal(D, 0.0)
        return D


# --------------------------------------------------------------------------------------
# Helpers (analytic single-qubit reference)
# --------------------------------------------------------------------------------------
_SIGMA_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
_SIGMA_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
_SIGMA_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.float64)
_I2      = np.eye(2, dtype=np.float64)

def single_qubit_expected(omega, delta, phi):
    # H = (Ω/2)(cosφ σx + sinφ σy) - Δ n,  with n=(I-σz)/2
    term_drive = (omega / 2.0) * (np.cos(phi) * _SIGMA_X + np.sin(phi) * _SIGMA_Y)
    n_op = 0.5 * (_I2 - _SIGMA_Z)
    return term_drive - delta * n_op


# --------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------

def test_single_qubit_dense_matches_analytic():
    reg = FakeRegister(np.array([[0.0]]))  # 1 site in 1D
    C6 = 0.0
    Omega = 3.2
    Delta = 1.1
    Phi = 0.4

    Hobj: RydbergHamiltonian = build_rydberg(
        reg, C6=C6, Omega=Omega, Delta=Delta, Phi=Phi
    )
    H = Hobj.to_dense()

    H_ref = single_qubit_expected(Omega, Delta, Phi)
    assert H.shape == (2, 2)
    assert np.allclose(H, H_ref, atol=1e-12)


def test_dtype_real_if_phi_zero_complex_if_phi_pi_over_2():
    reg = FakeRegister(np.array([[0.0], [1.0]]))  # 2 sites
    C6 = 0.0

    # All phases = 0 → purely real σx drive
    H_real = build_rydberg(reg, C6=C6, Omega=[1.0, 2.0], Delta=[0.0, 0.0], Phi=[0.0, 0.0]).to_dense()
    assert H_real.dtype == np.float64

    # All phases = π/2 → σy drive → imaginary off-diagonals → complex dtype
    H_cmplx = build_rydberg(reg, C6=C6, Omega=[1.0, 2.0], Delta=[0.0, 0.0], Phi=[np.pi/2, np.pi/2]).to_dense()
    # It should remain complex because σy contributes ±i explicitly
    assert np.iscomplexobj(H_cmplx)
    assert H_cmplx.dtype == np.complex128


def test_two_qubit_sparse_equals_dense_no_interactions():
    reg = FakeRegister(np.array([[0.0], [1.0]]))  # 2 sites
    C6 = 0.0
    Omega = [0.7, 0.3]
    Delta = [0.2, -0.4]
    Phi = [0.0, 0.0]  # purely real drive

    Hobj = build_rydberg(reg, C6=C6, Omega=Omega, Delta=Delta, Phi=Phi)
    Hd = Hobj.to_dense()
    Hs = Hobj.to_sparse()

    assert Hd.shape == (4, 4)
    assert Hs.shape == (4, 4)
    # Sparse should be real and match dense exactly
    assert Hs.dtype == np.float64
    assert np.allclose(Hd, Hs.toarray(), atol=1e-12)

    # Hermiticity
    assert np.allclose(Hd, Hd.conj().T, atol=1e-12)


def test_interaction_diagonal_matches_vij_n1n2_for_two_qubits():
    # Two sites at distance 1 → V_12 = C6 / 1^6 = C6
    reg = FakeRegister(np.array([[0.0], [1.0]]))
    C6 = 1.0
    Omega = [0.0, 0.0]
    Delta = [0.0, 0.0]
    Phi = [0.0, 0.0]

    H = build_rydberg(reg, C6=C6, Omega=Omega, Delta=Delta, Phi=Phi).to_dense()

    # Basis order is |00>, |01>, |10>, |11| (MSB=site0)
    diag = np.diag(H)
    # Only |11> has both n_i=1 → energy +V (here = +1)
    expected = np.array([0.0, 0.0, 0.0, 1.0])
    assert np.allclose(diag, expected, atol=1e-12)


def test_cutoff_zeroes_interaction_beyond_threshold():
    # Two sites at distance 1. If cutoff < 1 → interaction removed.
    reg = FakeRegister(np.array([[0.0], [1.0]]))
    C6 = 5.0
    cutoff = 0.5  # below inter-site distance

    H = build_rydberg(reg, C6=C6, Omega=0.0, Delta=0.0, Phi=0.0, cutoff=cutoff).to_dense()
    assert np.allclose(np.diag(H), 0.0, atol=1e-12)  # no interaction contributions


def test_bitflip_connectivity_matches_msb_convention_sparse():
    # N=2, drive on site 0 only (MSB), Φ=0 → σx on MSB flips:
    # |00> <-> |10>,  |01> <-> |11>
    reg = FakeRegister(np.array([[0.0], [1.0]]))
    C6 = 0.0
    Omega = [1.0, 0.0]  # only site 0 driven
    Delta = [0.0, 0.0]
    Phi = [0.0, 0.0]

    Hs = build_rydberg(reg, C6=C6, Omega=Omega, Delta=Delta, Phi=Phi).to_sparse()

    Hd = Hs.toarray()
    nz = np.argwhere(np.abs(Hd) > 1e-14)

    # Build a set of off-diagonal pairs with nonzero coupling
    offdiag_pairs = {(i, j) for i, j in map(tuple, nz) if i != j}

    # With MSB=site0, flips should connect (0<->2) and (1<->3)
    required_pairs = {(0, 2), (2, 0), (1, 3), (3, 1)}
    assert required_pairs.issubset(offdiag_pairs)

    # Ensure there is no drive flip on site 1 (LSB) (i.e., 0<->1 or 2<->3 absent)
    forbidden_pairs = {(0, 1), (1, 0), (2, 3), (3, 2)}
    assert offdiag_pairs.isdisjoint(forbidden_pairs)


def test_repr_and_metadata_fields():
    reg = FakeRegister(np.array([[0.0], [1.0], [2.0]]))
    Hobj = build_rydberg(reg, C6=0.0, Omega=[0, 0, 0], Delta=[0, 0, 0], Phi=[0, 0, 0])
    s = repr(Hobj)
    assert "RydbergHamiltonian" in s
    assert Hobj.num_sites == 3
    assert Hobj.hilbert_dim == 8
    assert Hobj.dtype == np.float64


def test_invalid_register_size_raises():
    # Register length inferred from coords mismatch caught in .from_register guard
    with pytest.raises(ValueError):
        bad_reg = FakeRegister(np.array([]).reshape(0, 1))  # N=0
        _ = build_rydberg(bad_reg, C6=0.0, Omega=0.0, Delta=0.0, Phi=0.0)