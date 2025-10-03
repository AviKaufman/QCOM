# tests/test_atoms.py
import numpy as np
import pytest

# Use a non-interactive backend for CI / headless runs
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qcom.atoms import AtomRegister  # if running inside repo: from atoms import AtomRegister


# ----------------------------- Fixtures -----------------------------

@pytest.fixture
def empty_reg():
    """An empty AtomRegister with shape (0, 3)."""
    return AtomRegister()

@pytest.fixture
def simple_reg():
    """Three coplanar atoms (z=0) for 2D paths."""
    return AtomRegister([
        (0.0,     0.0,     0.0),
        (1.0e-6,  0.0,     0.0),
        (0.0,     2.0e-6,  0.0),
    ])

@pytest.fixture
def three_d_reg():
    """Three atoms with nonzero z for 3D plotting paths."""
    return AtomRegister([
        (0.0,     0.0,     0.0),
        (1.0e-6,  0.0,     1.0e-6),
        (0.0,     2.0e-6,  2.0e-6),
    ])

# ----------------------------- __init__ -----------------------------

def test_init_empty_shape(empty_reg):
    # there should be no atoms in the empty register
    assert len(empty_reg) == 0
    # we should have a (0,3) array of positions as we have 0 atoms with 3 coords each
    assert empty_reg.positions.shape == (0, 3)
    # positions should be float64 dtype
    assert empty_reg.positions.dtype == np.float64


def test_init_with_positions_ok():
    # add a couple atoms
    reg = AtomRegister([(0, 0, 0), (1e-6, 0, 0)])
    # make sure we have only those atoms
    assert len(reg) == 2
    # positions should be (num_atoms, 3) = (2, 3)
    assert reg.positions.shape == (2, 3)
    # values preserved
    np.testing.assert_allclose(reg.positions[1], np.array([1e-6, 0.0, 0.0]))


def test_init_requires_2d_array_of_3_columns():
    # single tuple (shape (3,)) should fail (must be [(x,y,z)])
    with pytest.raises(ValueError):
        AtomRegister((0.0, 0.0, 0.0))  # not wrapped

    # wrong inner length of 1
    with pytest.raises(ValueError):
        AtomRegister([(0.0,)])  # only 1 coord

    # wrong inner length of 2
    with pytest.raises(ValueError):
        AtomRegister([(0.0, 0.0)])  # only 2 coords

    # wrong inner length of 4
    with pytest.raises(ValueError):
        AtomRegister([(0.0, 0.0, 0.0, 0.0)])  # has 4 coords

    # ragged inner lists 1
    with pytest.raises(ValueError):
        AtomRegister([(0.0, 0.0, 0.0), (1.0, 0.0)])  # second has 2

     # ragged inner lists 2
    with pytest.raises(ValueError):
        AtomRegister([(0.0, 0.0), (1.0, 0.0, 0.0)])  # first has 2
    
    # 1D single site array should fail (must be wrapped)
    with pytest.raises(ValueError):
        AtomRegister(np.array([0.0, 0.0, 0.0])) # input is an array instead of a list of tuples


def test_init_rejects_non_finite():
    # nan in any position should fail
    # x position
    with pytest.raises(ValueError):
        AtomRegister([(np.nan, 0.0, 0.0)])
    # y position
    with pytest.raises(ValueError):
        AtomRegister([(0.0, np.nan, 0.0)])
    # z position
    with pytest.raises(ValueError):
        AtomRegister([(0.0, 0.0, np.nan)])

    # inf in any position should fail
    # x position
    with pytest.raises(ValueError):
        AtomRegister([(np.inf, 0.0, 0.0)])
    # y position
    with pytest.raises(ValueError):
        AtomRegister([(0.0, np.inf, 0.0)])
    # z position
    with pytest.raises(ValueError):
        AtomRegister([(0.0, 0.0, np.inf)])

    # should also reject strings
    with pytest.raises(ValueError):
        AtomRegister([("a", 0.0, 0.0)])
    with pytest.raises(ValueError):
        AtomRegister([(0.0, "b", 0.0)])
    with pytest.raises(ValueError):
        AtomRegister([(0.0, 0.0, "c")])


def test_init_disallows_duplicates():
    # two identical atoms should fail
    with pytest.raises(ValueError):
        AtomRegister([(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)])

    # add several atoms before we repeat one
    with pytest.raises(ValueError):
        AtomRegister([
            (0.0, 0.0, 0.0),
            (1.0e-6, 0.0, 0.0),
            (0.0, 1.0e-6, 0.0),
            (1.0e-6, 1.0e-6, 0.0),
            (0.5e-6, 0.5e-6, 0.0),
            (1.0e-6, 0.5e-6, 0.0),
            (0.5e-6, 1.0e-6, 0.0),
            (1.0e-6, 1.0e-6, 0.0),  # duplicate of #4
        ])
    
def test_init_accepts_iterables():
    def gen():
        yield (0.0, 0.0, 0.0)
        yield (1e-6, 0.0, 0.0)
    reg = AtomRegister(gen())
    assert len(reg) == 2