import numpy as np
import pytest

from qcom.hamiltonians import IsingHamiltonian, build_ising
from qcom.lattice_register import LatticeRegister
from qcom.solvers.static import ground_state


def _reg(n: int) -> LatticeRegister:
    return LatticeRegister([(float(i), 0.0, 0.0) for i in range(n)])


def test_ising_scalar_coupling_nearest_neighbor_diagonal():
    H = build_ising(_reg(2), J=1.0, hx=0.0, hz=0.0)
    dense = H.to_dense()

    assert isinstance(H, IsingHamiltonian)
    assert dense.shape == (4, 4)
    assert np.allclose(np.diag(dense), [-1.0, 1.0, 1.0, -1.0])


def test_ising_vector_and_matrix_couplings_match_for_chain():
    reg = _reg(3)
    vector = build_ising(reg, J=[1.0, 2.0], hx=0.0, hz=0.3)
    matrix = build_ising(
        reg,
        J=np.array(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 2.0],
                [0.0, 2.0, 0.0],
            ]
        ),
        hx=0.0,
        hz=0.3,
    )

    assert np.allclose(vector.to_dense(), matrix.to_dense())


def test_ising_sparse_dense_and_matvec_equivalence():
    H = build_ising(_reg(3), J=0.7, hx=[0.1, 0.2, 0.3], hz=[-0.4, 0.0, 0.5])
    dense = H.to_dense()
    sparse = H.to_sparse()
    psi = np.arange(H.hilbert_dim, dtype=float)

    assert sparse.shape == dense.shape
    assert np.allclose(sparse.toarray(), dense)
    assert np.allclose(H.apply(psi), dense @ psi)
    assert np.allclose(H.to_linear_operator() @ psi, dense @ psi)


def test_ising_solver_compatibility():
    H = build_ising(_reg(2), J=1.0, hx=0.0, hz=0.0)
    e0 = ground_state(H, return_vector=False)

    assert np.isclose(e0, -1.0)


def test_ising_rejects_invalid_coupling_shapes():
    with pytest.raises(ValueError, match="length"):
        build_ising(_reg(3), J=[1.0])
    with pytest.raises(ValueError, match="NxN"):
        build_ising(_reg(3), J=np.ones((2, 2)))
