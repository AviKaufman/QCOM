import numpy as np
import pytest

# Use a non-interactive backend for CI / headless runs
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from qcom.lattice_register import LatticeRegister


# ----------------------------- Fixtures -----------------------------

@pytest.fixture
def empty_reg():
    """An empty LatticeRegister with shape (0, 3)."""
    return LatticeRegister()


@pytest.fixture
def simple_reg():
    """Three coplanar sites (z=0) for 2D paths."""
    return LatticeRegister([
        (0.0,     0.0,     0.0),
        (1.0e-6,  0.0,     0.0),
        (0.0,     2.0e-6,  0.0),
    ])


@pytest.fixture
def three_d_reg():
    """Three sites with nonzero z for 3D plotting paths."""
    return LatticeRegister([
        (0.0,     0.0,     0.0),
        (1.0e-6,  0.0,     1.0e-6),
        (0.0,     2.0e-6,  2.0e-6),
    ])


# ----------------------------- __init__ -----------------------------

def test_init_empty_shape(empty_reg):
    # there should be no sites in the empty register
    assert len(empty_reg) == 0
    # we should have a (0,3) array of positions as we have 0 sites with 3 coords each
    assert empty_reg.positions.shape == (0, 3)
    # positions should be float64 dtype
    assert empty_reg.positions.dtype == np.float64


def test_init_with_positions_ok():
    # add a couple sites
    reg = LatticeRegister([(0, 0, 0), (1e-6, 0, 0)])
    # make sure we have only those sites
    assert len(reg) == 2
    # positions should be (num_sites, 3) = (2, 3)
    assert reg.positions.shape == (2, 3)
    # values preserved
    np.testing.assert_allclose(reg.positions[1], np.array([1e-6, 0.0, 0.0]))


def test_init_requires_2d_array_of_3_columns():
    # single tuple (shape (3,)) should fail (must be [(x,y,z)])
    with pytest.raises(ValueError):
        LatticeRegister((0.0, 0.0, 0.0))  # not wrapped

    # wrong inner length of 1
    with pytest.raises(ValueError):
        LatticeRegister([(0.0,)])  # only 1 coord

    # wrong inner length of 2
    with pytest.raises(ValueError):
        LatticeRegister([(0.0, 0.0)])  # only 2 coords

    # wrong inner length of 4
    with pytest.raises(ValueError):
        LatticeRegister([(0.0, 0.0, 0.0, 0.0)])  # has 4 coords

    # ragged inner lists 1
    with pytest.raises(ValueError):
        LatticeRegister([(0.0, 0.0, 0.0), (1.0, 0.0)])  # second has 2

    # ragged inner lists 2
    with pytest.raises(ValueError):
        LatticeRegister([(0.0, 0.0), (1.0, 0.0, 0.0)])  # first has 2

    # 1D single site array should fail (must be wrapped)
    with pytest.raises(ValueError):
        LatticeRegister(np.array([0.0, 0.0, 0.0]))  # input is an array instead of a list of tuples


def test_init_rejects_non_finite():
    # nan in any position should fail
    # x position
    with pytest.raises(ValueError):
        LatticeRegister([(np.nan, 0.0, 0.0)])
    # y position
    with pytest.raises(ValueError):
        LatticeRegister([(0.0, np.nan, 0.0)])
    # z position
    with pytest.raises(ValueError):
        LatticeRegister([(0.0, 0.0, np.nan)])

    # inf in any position should fail
    # x position
    with pytest.raises(ValueError):
        LatticeRegister([(np.inf, 0.0, 0.0)])
    # y position
    with pytest.raises(ValueError):
        LatticeRegister([(0.0, np.inf, 0.0)])
    # z position
    with pytest.raises(ValueError):
        LatticeRegister([(0.0, 0.0, np.inf)])

    # should also reject strings
    with pytest.raises(ValueError):
        LatticeRegister([("a", 0.0, 0.0)])
    with pytest.raises(ValueError):
        LatticeRegister([(0.0, "b", 0.0)])
    with pytest.raises(ValueError):
        LatticeRegister([(0.0, 0.0, "c")])


def test_init_disallows_duplicates():
    # two identical sites should fail
    with pytest.raises(ValueError):
        LatticeRegister([(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)])

    # add several sites before we repeat one
    with pytest.raises(ValueError):
        LatticeRegister([
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
    reg = LatticeRegister(gen())
    assert len(reg) == 2


# ----------------------------- __len__ -----------------------------

def test_len_tracks_number_of_sites(empty_reg):
    empty_reg.add((0.0, 0.0, 0.0))
    empty_reg.add((1.0e-6, 0.0, 0.0))
    assert len(empty_reg) == 2


# ----------------------------- positions property -----------------------------

def test_positions_view_is_read_only(simple_reg):
    view = simple_reg.positions
    assert view.flags["WRITEABLE"] is False
    with pytest.raises(ValueError):
        view[0, 0] = 123.0


# ----------------------------- add() -----------------------------

def test_add_appends_insertion_order(empty_reg):
    idx0 = empty_reg.add((0.0, 0.0, 0.0))
    idx1 = empty_reg.add((1.0e-6, 0.0, 0.0))
    idx2 = empty_reg.add((0.0, 1.0e-6, 0.0))
    assert (idx0, idx1, idx2) == (0, 1, 2)
    np.testing.assert_allclose(empty_reg.positions[1], np.array([1.0e-6, 0.0, 0.0]))


def test_add_requires_three_coordinates(empty_reg):
    with pytest.raises(ValueError):
        empty_reg.add((0.0, 0.0))  # too short
    with pytest.raises(ValueError):
        empty_reg.add((0.0, 0.0, 0.0, 0.0))  # too long


def test_add_rejects_duplicate_sites(empty_reg):
    empty_reg.add((0.0, 0.0, 0.0))
    with pytest.raises(ValueError):
        empty_reg.add((0.0, 0.0, 0.0))


def test_add_rejects_non_finite_values(empty_reg):
    with pytest.raises(ValueError):
        empty_reg.add((np.nan, 0.0, 0.0))


# ----------------------------- remove() -----------------------------

def test_remove_by_index(simple_reg):
    removed = simple_reg.remove(1)
    assert removed == 1
    assert len(simple_reg) == 2
    with pytest.raises(IndexError):
        simple_reg.position(2)


def test_remove_invalid_index(simple_reg):
    with pytest.raises(IndexError):
        simple_reg.remove(10)


# ----------------------------- position() -----------------------------

def test_position_returns_coordinates(simple_reg):
    pos = simple_reg.position(1)
    assert pos == (1.0e-6, 0.0, 0.0)


def test_position_invalid_index(simple_reg):
    with pytest.raises(IndexError):
        simple_reg.position(-1)
    with pytest.raises(IndexError):
        simple_reg.position(99)


# ----------------------------- as_array() -----------------------------

def test_as_array_returns_independent_copy(simple_reg):
    arr = simple_reg.as_array()
    arr[0, 0] = 123.0
    assert simple_reg.positions[0, 0] == 0.0


# ----------------------------- index_map() -----------------------------

def test_index_map_formats_expected_rows(simple_reg):
    map_str = simple_reg.index_map()
    lines = map_str.splitlines()
    assert lines[0].startswith("index  x (m)")
    assert lines[2].strip().startswith("0")
    assert "0.000000e+00" in lines[2]
    assert "1.000000e-06" in lines[3]


def test_index_map_respects_max_rows(simple_reg):
    map_str = simple_reg.index_map(max_rows=2)
    assert "... (" in map_str


# ----------------------------- __repr__ -----------------------------

def test_repr_contains_summary(simple_reg):
    rep = repr(simple_reg)
    assert "LatticeRegister" in rep
    assert "N=3" in rep


# ----------------------------- distance() -----------------------------

def test_distance_between_sites(simple_reg):
    dist = simple_reg.distance(0, 1)
    assert dist == pytest.approx(1.0e-6)


def test_distance_same_site_is_zero(simple_reg):
    assert simple_reg.distance(0, 0) == pytest.approx(0.0)


def test_distance_invalid_indices(simple_reg):
    with pytest.raises(IndexError):
        simple_reg.distance(0, 99)


# ----------------------------- distances() -----------------------------

def test_distances_matrix_is_symmetric(simple_reg):
    D = simple_reg.distances()
    assert D.shape == (3, 3)
    np.testing.assert_allclose(D, D.T)


def test_distances_diagonal_is_zero(simple_reg):
    D = simple_reg.distances()
    np.testing.assert_allclose(np.diag(D), np.zeros(3))


# ----------------------------- clear() -----------------------------

def test_clear_empties_register(simple_reg):
    result = simple_reg.clear()
    assert result is None
    assert len(simple_reg) == 0
    assert simple_reg.positions.shape == (0, 3)


# ----------------------------- plot() -----------------------------

def test_plot_returns_2d_axes(simple_reg):
    ax = simple_reg.plot(show_index=False)
    try:
        assert isinstance(ax, matplotlib.axes.Axes)
        assert not isinstance(ax, Axes3D)
    finally:
        plt.close(ax.figure)


def test_plot_returns_3d_axes(three_d_reg):
    ax = three_d_reg.plot(show_index=False)
    try:
        assert isinstance(ax, Axes3D)
    finally:
        plt.close(ax.figure)


def test_plot_handles_empty_register(empty_reg):
    ax = empty_reg.plot(show_index=False)
    try:
        assert isinstance(ax, matplotlib.axes.Axes)
    finally:
        plt.close(ax.figure)
