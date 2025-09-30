import pytest
import numpy as np
import matplotlib

# use a non-interactive backend so tests don’t pop up windows
matplotlib.use("Agg")
import qcom as qc
from qcom.atoms import AtomRegister


def test_init_valid_and_len():
    pos = [(0, 0), (1.5, -2.0), (3, 3)]
    reg = AtomRegister(pos)
    # length
    assert len(reg) == 3
    # positions stored correctly
    np.testing.assert_allclose(reg.positions, np.array(pos, dtype=float))


def test_init_invalid_shape_raises():
    # wrong inner dimensions
    with pytest.raises(ValueError):
        AtomRegister([1, 2, 3])  # not pairs
    with pytest.raises(ValueError):
        AtomRegister([(0, 0, 0), (1, 1, 1)])  # 3-tuples


def test_chain_factory():
    reg = AtomRegister.chain(4, a=2.0, y=1.0)
    # should have 4 atoms
    assert len(reg) == 4
    expected = np.array([[0, 1.0], [2, 1.0], [4, 1.0], [6, 1.0]])
    np.testing.assert_allclose(reg.positions, expected)


def test_ladder_factory():
    reg = AtomRegister.ladder(ncol=2, a=1.0, rho=3.0, y0=-1.0)
    # 2 columns × 2 legs = 4 atoms
    assert len(reg) == 4
    expected = np.array(
        [
            [0.0, -1.0],
            [0.0, 2.0],  # -1 + 3
            [1.0, -1.0],
            [1.0, 2.0],
        ]
    )
    np.testing.assert_allclose(reg.positions, expected)


def test_grid_factory():
    reg = AtomRegister.grid(nx=2, ny=3, a=0.5, x0=1.0, y0=-1.0)
    # 2×3 = 6 atoms
    assert len(reg) == 6
    # check a few corner positions
    corners = {
        (0, 0): (1.0, -1.0),
        (1, 0): (1.5, -1.0),
        (0, 2): (1.0, 0.0),
        (1, 2): (1.5, 0.0),
    }
    for (i, j), coord in corners.items():
        idx = i * 3 + j
        np.testing.assert_allclose(reg.positions[idx], coord)


def test_add_and_invalid_add():
    reg = AtomRegister.chain(2, a=1.0)
    reg.add((5.5, -2.2))
    assert len(reg) == 3
    # last point is the one we just added
    np.testing.assert_allclose(reg.positions[-1], [5.5, -2.2])
    # invalid additions
    with pytest.raises(ValueError):
        reg.add((1, 2, 3))  # wrong length
    with pytest.raises(ValueError):
        reg.add("not a tup")  # wrong type


def test_remove_and_invalid_remove():
    reg = AtomRegister.chain(3, a=1.0)
    reg.remove(1)
    assert len(reg) == 2
    # removing out of range
    with pytest.raises(IndexError):
        reg.remove(5)
    with pytest.raises(IndexError):
        reg.remove(-10)


def test_translate():
    reg = AtomRegister.chain(2, a=2.0)
    before = reg.positions.copy()
    reg.translate(dx=1.0, dy=-0.5)
    np.testing.assert_allclose(reg.positions, before + np.array([1.0, -0.5]))


def test_distance_and_distances_matrix():
    # make a simple square
    pos = [(0, 0), (1, 0), (1, 1), (0, 1)]
    reg = AtomRegister(pos)
    # known distances
    assert pytest.approx(reg.distance(0, 1)) == 1.0
    assert pytest.approx(reg.distance(0, 2)) == np.sqrt(2)
    mat = reg.distances()
    # symmetric and zeros on diagonal
    assert mat.shape == (4, 4)
    assert np.allclose(mat, mat.T)
    assert np.all(mat.diagonal() == 0)


def test_extend_and_plus_operator():
    r1 = AtomRegister.chain(2, a=1.0)
    r2 = AtomRegister.chain(1, a=1.0, y=5.0)
    # extend in-place
    r1.extend(r2)
    assert len(r1) == 3
    # plus operator returns new register
    r3 = AtomRegister.chain(2, a=1.0) + AtomRegister.chain(3, a=1.0, y=2.0)
    assert isinstance(r3, AtomRegister)
    assert len(r3) == 5


def test_plot_returns_axis_and_points():
    reg = AtomRegister.chain(3, a=0.3)
    ax = reg.plot()
    # ensure we got an Axes back
    assert hasattr(ax, "collections")
    # scatter created exactly one Collection
    assert len(ax.collections) == 1
    offsets = ax.collections[0].get_offsets()
    # number of plotted points = number of atoms
    assert offsets.shape[0] == len(reg)


if __name__ == "__main__":
    # Run the tests from only this file
    pytest.main([__file__])
