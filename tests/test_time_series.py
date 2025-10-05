# tests/test_time_series.py
import numpy as np
import pytest

# If installed as a package:
from qcom.time_series import TimeSeries
# If running inside the repo instead, use:
# from time_series import TimeSeries

# ----------------------------- Fixtures -----------------------------

@pytest.fixture
def ts_multi():
    # Omega over [0, 2e-6], Delta over [-1e-6, 0], Phi single point at 2e-6
    return TimeSeries(
        Omega=([0.0, 1e-6, 2e-6], [0.0, 0.5, 1.0]),
        Delta=([-1e-6, 0.0],      [-0.5, 0.0]),
        Phi=([2e-6],             [3.14159]),
    )

@pytest.fixture
def ts_empty():
    return TimeSeries()

@pytest.fixture
def empty_abs():
    return TimeSeries(mode="absolute")

@pytest.fixture
def empty_norm():
    return TimeSeries(mode="normalized")

@pytest.fixture
def base_abs():
    """Absolute-mode series with one channel 'Omega' at 3 times."""
    ts = TimeSeries(mode="absolute")
    ts.add_series("Omega", [0.0, 1.0, 2.0], [0.0, 1.0, 2.0])
    return ts

@pytest.fixture
def base_norm():
    """Normalized-mode series with typical ranges."""
    ts = TimeSeries(mode="normalized")
    ts.add_series("Omega", [0.0, 1.0], [0.0, 1.0])
    ts.add_series("Delta", [0.0, 1.0, 2.0], [-1.0, 0.0, 1.0])
    return ts

@pytest.fixture
def ts_two_channels_abs():
    """Absolute-mode with staggered starts and different lengths."""
    ts = TimeSeries(mode="absolute")
    ts.add_series("Omega", [0.0, 1.0, 2.0], [0.0, 0.5, 1.0])
    ts.add_series("Delta", [0.5, 2.0], [10.0, 12.0])
    return ts


@pytest.fixture
def ts_mixed_negative():
    """Absolute-mode with one channel starting negative."""
    ts = TimeSeries(mode="absolute")
    ts.add_series("Omega", [1.0, 2.0], [5.0, 6.0])     # starts at +1
    ts.add_series("Delta", [-3.0, -1.0], [7.0, 8.0])   # starts at -3 (earliest)
    return ts

@pytest.fixture
def ts_abs():
    ts = TimeSeries(mode="absolute")
    ts.add_series("Omega", [0.0, 1.0, 2.0], [0.0,  1.0,  0.0])
    ts.add_series("Delta", [0.5, 1.5],       [10.0, 12.0])
    return ts

@pytest.fixture
def ts_norm():
    ts = TimeSeries(mode="normalized")
    ts.add_series("Omega", [0.0, 1.0], [0.0, 1.0])   # in [0,1]
    ts.add_series("Delta", [0.0, 1.0], [-1.0, 1.0])  # in [-1,1]
    return ts


# ----------------------------- __init__ (constructor) -----------------------------

def test_init_empty_ok():
    ts = TimeSeries()
    assert ts.mode == "absolute"
    assert hasattr(ts, "_channels")
    assert isinstance(ts._channels, dict)
    assert len(ts._channels) == 0  # empty series


def test_init_rejects_bad_mode():
    with pytest.raises(ValueError):
        TimeSeries(mode="weird")


def test_init_with_one_channel_ok_absolute():
    t = [0.0, 1.0e-6, 2.0e-6]
    v = [0.0, 0.5, 1.0]
    ts = TimeSeries(Omega=(t, v))
    assert "Omega" in ts._channels
    t_stored, v_stored = ts._channels["Omega"]
    np.testing.assert_allclose(t_stored, np.asarray(t, dtype=np.float64))
    np.testing.assert_allclose(v_stored, np.asarray(v, dtype=np.float64))
    assert t_stored.dtype == np.float64 and v_stored.dtype == np.float64
    assert t_stored.flags["C_CONTIGUOUS"] and v_stored.flags["C_CONTIGUOUS"]


def test_init_requires_1d_equal_length():
    # 2D times
    with pytest.raises(ValueError):
        TimeSeries(Omega=(np.array([[0.0, 1.0]]), np.array([0.0, 1.0])))

    # length mismatch
    with pytest.raises(ValueError):
        TimeSeries(Omega=([0.0, 1.0], [0.0]))


def test_init_requires_nonempty_channel():
    with pytest.raises(ValueError):
        TimeSeries(Omega=([], []))


def test_init_requires_strictly_increasing_times():
    # duplicate time
    with pytest.raises(ValueError):
        TimeSeries(Omega=([0.0, 1.0e-6, 1.0e-6], [0.0, 0.1, 0.2]))

    # decreasing
    with pytest.raises(ValueError):
        TimeSeries(Omega=([0.0, -1.0e-6], [0.0, 1.0]))


def test_init_rejects_nonfinite_times_and_values():
    # NaN time
    with pytest.raises(ValueError):
        TimeSeries(Omega=([0.0, np.nan], [0.0, 1.0]))
    # Inf time
    with pytest.raises(ValueError):
        TimeSeries(Omega=([0.0, np.inf], [0.0, 1.0]))
    # NaN value
    with pytest.raises(ValueError):
        TimeSeries(Omega=([0.0, 1.0], [0.0, np.nan]))
    # Inf value
    with pytest.raises(ValueError):
        TimeSeries(Omega=([0.0, 1.0], [0.0, np.inf]))


def test_init_allows_negative_times():
    # Negative times are allowed as long as strictly increasing
    ts = TimeSeries(Delta=([-2.0e-6, -1.0e-6, 0.0], [0.0, -0.2, -0.4]))
    assert "Delta" in ts._channels
    t, _ = ts._channels["Delta"]
    assert t[0] < 0.0 and t[-1] == 0.0


# ----------------------------- normalized-mode bounds -----------------------------

def test_init_normalized_omega_bounds_enforced():
    # Edges OK
    TimeSeries(mode="normalized", Omega=([0.0, 1.0], [0.0, 1.0]))
    # Below 0
    with pytest.raises(ValueError):
        TimeSeries(mode="normalized", Omega=([0.0, 1.0], [-1e-3, 0.1]))
    # Above 1
    with pytest.raises(ValueError):
        TimeSeries(mode="normalized", Omega=([0.0, 1.0], [0.5, 1.001]))


def test_init_normalized_delta_bounds_enforced():
    # Edges OK
    TimeSeries(mode="normalized", Delta=([0.0, 1.0], [-1.0, 1.0]))
    # Below -1
    with pytest.raises(ValueError):
        TimeSeries(mode="normalized", Delta=([0.0, 1.0], [-1.001, 0.0]))
    # Above 1
    with pytest.raises(ValueError):
        TimeSeries(mode="normalized", Delta=([0.0, 1.0], [0.0, 1.01]))


def test_init_normalized_phi_unbounded():
    # Phi has no bounds in normalized mode
    ts = TimeSeries(mode="normalized", Phi=([0.0, 1.0], [10.0, -123.456]))
    assert "Phi" in ts._channels

# ----------------------------- channels (read-only mapping of read-only arrays) -----------------------------

def test_channels_returns_readonly_mapping_and_views():
    t = [0.0, 1e-6, 2e-6]
    v = [0.0, 0.4, 0.9]
    ts = TimeSeries(Omega=(t, v))

    ch = ts.channels  # MappingProxyType
    # Mapping is read-only
    with pytest.raises(TypeError):
        ch["New"] = (np.array([0.0]), np.array([0.0]))

    # Access a channel and ensure the arrays are read-only views
    t_view, v_view = ch["Omega"]
    assert isinstance(t_view, np.ndarray) and isinstance(v_view, np.ndarray)
    assert t_view.flags.writeable is False
    assert v_view.flags.writeable is False
    np.testing.assert_allclose(t_view, np.asarray(t, dtype=np.float64))
    np.testing.assert_allclose(v_view, np.asarray(v, dtype=np.float64))

    # Attempting to mutate the views must fail
    with pytest.raises(ValueError):
        t_view[0] = 123.0
    with pytest.raises(ValueError):
        v_view[-1] = -5.0


def test_channels_empty_mapping_when_no_channels():
    ts = TimeSeries()
    ch = ts.channels
    assert len(ch) == 0
    # Still read-only
    with pytest.raises(TypeError):
        ch["Omega"] = (np.array([0.0]), np.array([0.0]))


# ----------------------------- channel_names (order and content) -----------------------------

def test_channel_names_reflect_insertion_order():
    # Insert Delta first, then Omega; dicts preserve insertion order in Py3.7+
    ts = TimeSeries(
        Delta=([-1e-6, 0.0, 1e-6], [-0.5, 0.0, 0.5]),
        Omega=([0.0, 1e-6], [0.0, 1.0]),
    )
    names = ts.channel_names
    assert names == ("Delta", "Omega")

    # Single-channel case
    ts2 = TimeSeries(Phi=([0.0], [3.14]))
    assert ts2.channel_names == ("Phi",)

    # Empty case
    ts3 = TimeSeries()
    assert ts3.channel_names == tuple()


# ----------------------------- is_empty -----------------------------

def test_is_empty_true_for_empty_false_when_channel_present():
    ts = TimeSeries()
    assert ts.is_empty is True

    ts2 = TimeSeries(Omega=([0.0], [0.0]))
    assert ts2.is_empty is False

# ----------------------------- __len__ -----------------------------

def test_len_counts_channels(ts_multi, ts_empty):
    assert len(ts_empty) == 0
    assert len(ts_multi) == 3


# ----------------------------- has_channel -----------------------------

def test_has_channel_true_false(ts_multi):
    assert ts_multi.has_channel("Omega") is True
    assert ts_multi.has_channel("Delta") is True
    assert ts_multi.has_channel("Phi")   is True
    assert ts_multi.has_channel("Nope")  is False


# ----------------------------- times(name) -----------------------------

def test_times_returns_readonly_view_and_matches(ts_multi):
    t = ts_multi.times("Omega")
    assert isinstance(t, np.ndarray)
    assert t.flags.writeable is False
    np.testing.assert_allclose(t, np.array([0.0, 1e-6, 2e-6], dtype=np.float64))

def test_times_raises_for_missing_channel(ts_multi):
    with pytest.raises(KeyError):
        ts_multi.times("NoSuchChannel")


# ----------------------------- values(name) -----------------------------

def test_values_returns_readonly_view_and_matches(ts_multi):
    v = ts_multi.values("Delta")
    assert isinstance(v, np.ndarray)
    assert v.flags.writeable is False
    np.testing.assert_allclose(v, np.array([-0.5, 0.0], dtype=np.float64))

def test_values_raises_for_missing_channel(ts_multi):
    with pytest.raises(KeyError):
        ts_multi.values("NoSuchChannel")


# ----------------------------- domain(name=None) -----------------------------

def test_domain_per_channel(ts_multi):
    # Exact per-channel domains
    assert ts_multi.domain("Omega") == (0.0, 2e-6)
    assert ts_multi.domain("Delta") == (-1e-6, 0.0)
    assert ts_multi.domain("Phi")   == (2e-6, 2e-6)  # single point domain

def test_domain_union(ts_multi):
    # Union should span min(firsts) to max(lasts)
    tmin, tmax = ts_multi.domain()
    assert tmin == -1e-6
    assert tmax ==  2e-6

def test_domain_raises_on_missing_channel(ts_multi):
    with pytest.raises(KeyError):
        ts_multi.domain("NoSuchChannel")

def test_domain_raises_when_empty(ts_empty):
    with pytest.raises(ValueError):
        ts_empty.domain()

# ----------------------------- __repr__ -----------------------------

def test_repr_empty():
    ts = TimeSeries()
    s = repr(ts)
    assert "TimeSeries(" in s
    assert "mode='absolute'" in s
    assert "channels=∅" in s

def test_repr_single_channel_single_point():
    ts = TimeSeries(Omega=([1.0], [0.5]))
    s = repr(ts)
    # Example: "Omega[L=1, 1.000e+00→1.000e+00s]"
    assert "Omega[L=1" in s
    assert "→" in s
    assert "s]" in s

def test_repr_multiple_channels_order_is_insertion_order():
    ts = TimeSeries(Delta=([0.0, 1.0], [0.0, 0.1]), Omega=([0.0], [1.0]))
    s = repr(ts)
    # Should list Delta before Omega based on kwargs insertion
    assert s.index("Delta[L=2") < s.index("Omega[L=1")


# ----------------------------- _rel_tol -----------------------------

def test_rel_tol_small_and_large():
    # Static method behavior: 1e-12 * max(1, |t|)
    assert TimeSeries._rel_tol(0.0) == pytest.approx(1e-12)
    assert TimeSeries._rel_tol(1e-9) == pytest.approx(1e-12)  # still uses 1
    assert TimeSeries._rel_tol(2.0) == pytest.approx(2.0e-12)
    assert TimeSeries._rel_tol(-3.0) == pytest.approx(3.0e-12)
    # Non-float inputs should be cast safely
    assert TimeSeries._rel_tol(np.float64(5.0)) == pytest.approx(5e-12)


# ----------------------------- _check_normalized_bounds -----------------------------

def test_check_normalized_bounds_noop_in_absolute_mode():
    ts = TimeSeries(mode="absolute")
    # Should not raise for any channel/value
    ts._check_normalized_bounds("Omega", np.array([-10.0, 10.0]))
    ts._check_normalized_bounds("Delta", np.array([-2.0, 2.0]))
    ts._check_normalized_bounds("Phi",   np.array([1e6, -1e6]))

def test_check_normalized_bounds_omega_ok_and_violations():
    ts = TimeSeries(mode="normalized")
    # In-range OK
    ts._check_normalized_bounds("Omega", np.array([0.0, 0.5, 1.0]))
    ts._check_normalized_bounds("omega", np.array([0.25]))  # case-insensitive

    # Below 0
    with pytest.raises(ValueError):
        ts._check_normalized_bounds("Omega", np.array([-1e-9]))
    # Above 1
    with pytest.raises(ValueError):
        ts._check_normalized_bounds("OMEGA", np.array([1.0000000001]))

def test_check_normalized_bounds_delta_ok_and_violations():
    ts = TimeSeries(mode="normalized")
    # In-range OK
    ts._check_normalized_bounds("Delta", np.array([-1.0, 0.0, 1.0]))
    ts._check_normalized_bounds("delta", np.array([-0.1, 0.2]))

    # Below -1
    with pytest.raises(ValueError):
        ts._check_normalized_bounds("Delta", np.array([-1.0000001]))
    # Above 1
    with pytest.raises(ValueError):
        ts._check_normalized_bounds("DELTA", np.array([1.0000001]))

def test_check_normalized_bounds_phi_unbounded():
    ts = TimeSeries(mode="normalized")
    # Phi is intentionally unbounded in normalized mode
    ts._check_normalized_bounds("Phi", np.array([-1e9, 0.0, 1e9]))

# ----------------------------- add_point -----------------------------

def test_add_point_creates_channel_and_returns_index(empty_abs):
    ts = empty_abs
    idx0 = ts.add_point("Omega", 0.0, 1.0)
    assert idx0 == 0
    assert ts.has_channel("Omega")
    np.testing.assert_allclose(ts.times("Omega"), [0.0])
    np.testing.assert_allclose(ts.values("Omega"), [1.0])

def test_add_point_inserts_sorted_and_returns_position(empty_abs):
    ts = empty_abs
    ts.add_point("Omega", 1.0, 10.0)   # [1.0]
    ts.add_point("Omega", 0.0,  0.0)   # insert left -> [0.0, 1.0]
    idx = ts.add_point("Omega", 0.5,  5.0)   # middle -> [0.0, 0.5, 1.0]
    assert idx == 1
    np.testing.assert_allclose(ts.times("Omega"), [0.0, 0.5, 1.0])
    np.testing.assert_allclose(ts.values("Omega"), [0.0, 5.0, 10.0])

def test_add_point_replaces_within_default_tol(empty_abs):
    ts = empty_abs
    ts.add_point("Omega", 1.0, 10.0)
    # default tol = 1e-12 * max(1, |t|)
    # choose dt << tol to force replacement
    idx = ts.add_point("Omega", 1.0 + 1e-15, 20.0)
    assert idx == 0
    np.testing.assert_allclose(ts.times("Omega"), [1.0])
    np.testing.assert_allclose(ts.values("Omega"), [20.0])

def test_add_point_explicit_tol_controls_replacement(empty_abs):
    ts = empty_abs
    ts.add_point("Omega", 1.0, 1.0)
    # Set tiny tolerance, so 1.0 + 1e-15 is NOT considered equal and should insert
    idx = ts.add_point("Omega", 1.0 + 1e-15, 2.0, tol=1e-16)
    assert idx == 1
    np.testing.assert_allclose(ts.times("Omega"), [1.0, 1.0 + 1e-15])
    np.testing.assert_allclose(ts.values("Omega"), [1.0, 2.0])

def test_add_point_nonfinite_raises(empty_abs):
    ts = empty_abs
    with pytest.raises(ValueError):
        ts.add_point("Omega", np.nan, 1.0)
    with pytest.raises(ValueError):
        ts.add_point("Omega", 0.0, np.inf)

def test_add_point_normalized_bounds_omega(empty_norm):
    ts = empty_norm
    ts.add_point("Omega", 0.0, 0.0)  # ok
    ts.add_point("Omega", 1.0, 1.0)  # ok
    with pytest.raises(ValueError):
        ts.add_point("Omega", 2.0, -1e-9)  # < 0
    with pytest.raises(ValueError):
        ts.add_point("Omega", 3.0, 1.0000001)  # > 1

def test_add_point_normalized_bounds_delta(empty_norm):
    ts = empty_norm
    ts.add_point("Delta", 0.0, -1.0)  # ok
    ts.add_point("Delta", 1.0,  1.0)  # ok
    with pytest.raises(ValueError):
        ts.add_point("Delta", 2.0, -1.000001)
    with pytest.raises(ValueError):
        ts.add_point("Delta", 3.0,  1.000001)

def test_add_point_case_insensitive_channel_norm_checks(empty_norm):
    ts = empty_norm
    ts.add_point("oMeGa", 0.0, 0.5)  # ok
    with pytest.raises(ValueError):
        ts.add_point("OMEGA", 1.0, 1.1)


# ----------------------------- add_series -----------------------------

def test_add_series_creates_channel_sorted_and_dedup(empty_abs):
    ts = empty_abs
    # Unsorted with duplicate times; last value wins for duplicates
    L = ts.add_series("Omega",
                      times=[0.2, 0.1, 0.2, 0.0],
                      values=[2.0, 1.0, 3.0, 0.0])
    assert L == 3
    np.testing.assert_allclose(ts.times("Omega"),  [0.0, 0.1, 0.2])
    np.testing.assert_allclose(ts.values("Omega"), [0.0, 1.0, 3.0])

def test_add_series_returns_existing_length_if_empty_input(empty_abs):
    ts = empty_abs
    assert ts.add_series("Omega", [], []) == 0
    ts.add_point("Omega", 1.0, 7.0)
    assert ts.add_series("Omega", [], []) == 1

def test_add_series_nonfinite_raises(empty_abs):
    ts = empty_abs
    with pytest.raises(ValueError):
        ts.add_series("Omega", [0.0, np.nan], [1.0, 2.0])
    with pytest.raises(ValueError):
        ts.add_series("Omega", [0.0, 1.0], [1.0, np.inf])

def test_add_series_shape_mismatch_raises(empty_abs):
    ts = empty_abs
    with pytest.raises(ValueError):
        ts.add_series("Omega", [0.0], [1.0, 2.0])

def test_add_series_normalized_bounds_checked_batch(empty_norm):
    ts = empty_norm
    with pytest.raises(ValueError):
        ts.add_series("Omega", [0.0, 1.0], [-0.1, 0.1])  # < 0
    with pytest.raises(ValueError):
        ts.add_series("Delta", [0.0, 1.0], [-1.1, 0.0])  # < -1

def test_add_series_append_fast_path(empty_abs):
    ts = empty_abs
    ts.add_series("Omega", [0.0, 1.0], [0.0, 1.0])
    # All new times strictly after existing -> append path
    L = ts.add_series("Omega", [2.0, 3.0], [2.0, 3.0])
    assert L == 4
    np.testing.assert_allclose(ts.times("Omega"),  [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_allclose(ts.values("Omega"), [0.0, 1.0, 2.0, 3.0])

def test_add_series_prepend_fast_path(empty_abs):
    ts = empty_abs
    ts.add_series("Omega", [2.0, 3.0], [2.0, 3.0])
    # All new times strictly before existing -> prepend path
    L = ts.add_series("Omega", [0.0, 1.0], [0.0, 1.0])
    assert L == 4
    np.testing.assert_allclose(ts.times("Omega"),  [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_allclose(ts.values("Omega"), [0.0, 1.0, 2.0, 3.0])

def test_add_series_general_merge_with_replacement(empty_abs):
    ts = empty_abs
    ts.add_series("Omega", [0.0, 1.0, 3.0], [0.0, 1.0, 3.0])  # old
    # new overlaps at t=1.0 (replace), inserts at 2.0, and duplicates at 3.0 (replace)
    L = ts.add_series("Omega", [1.0, 2.0, 3.0], [10.0, 20.0, 30.0])
    assert L == 4
    np.testing.assert_allclose(ts.times("Omega"),  [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_allclose(ts.values("Omega"), [0.0, 10.0, 20.0, 30.0])

def test_add_series_duplicate_collapse_in_incoming_last_wins(empty_abs):
    ts = empty_abs
    ts.add_series("Omega", [0.0], [0.0])
    # Incoming has 0.5 three times; last value wins before merge
    L = ts.add_series("Omega", [0.5, 0.2, 0.5, 0.5], [5.0, 2.0, 6.0, 7.0])
    assert L == 3
    np.testing.assert_allclose(ts.times("Omega"),  [0.0, 0.2, 0.5])
    np.testing.assert_allclose(ts.values("Omega"), [0.0, 2.0, 7.0])

def test_add_series_tolerance_controls_duplicate_vs_insert(empty_abs):
    ts = empty_abs
    ts.add_series("Omega", [1.0], [1.0])
    # By default, 1.0 + 1e-15 is within default rtol -> replacement
    L1 = ts.add_series("Omega", [1.0 + 1e-15], [2.0])
    assert L1 == 1
    np.testing.assert_allclose(ts.times("Omega"),  [1.0])
    np.testing.assert_allclose(ts.values("Omega"), [2.0])
    # With tiny explicit tol, treat as new sample (insert)
    L2 = ts.add_series("Omega", [1.0 + 1e-15], [3.0], tol=1e-16)
    assert L2 == 2
    np.testing.assert_allclose(ts.times("Omega"),  [1.0, 1.0 + 1e-15])
    np.testing.assert_allclose(ts.values("Omega"), [2.0, 3.0])

# ----------------------------- remove_point -----------------------------

def test_remove_point_exact_match_middle(base_abs):
    ts = base_abs
    # remove t=1.0 (middle)
    idx = ts.remove_point("Omega", 1.0)
    assert idx == 1
    np.testing.assert_allclose(ts.times("Omega"),  [0.0, 2.0])
    np.testing.assert_allclose(ts.values("Omega"), [0.0, 2.0])

def test_remove_point_exact_match_first(base_abs):
    ts = base_abs
    idx = ts.remove_point("Omega", 0.0)
    assert idx == 0
    np.testing.assert_allclose(ts.times("Omega"),  [1.0, 2.0])
    np.testing.assert_allclose(ts.values("Omega"), [1.0, 2.0])

def test_remove_point_exact_match_last(base_abs):
    ts = base_abs
    idx = ts.remove_point("Omega", 2.0)
    assert idx == 2
    np.testing.assert_allclose(ts.times("Omega"),  [0.0, 1.0])
    np.testing.assert_allclose(ts.values("Omega"), [0.0, 1.0])

def test_remove_point_within_default_tol_replaces_match(base_abs):
    ts = base_abs
    # default tol ~ 1e-12 * max(1, |t|)
    idx = ts.remove_point("Omega", 1.0 + 1e-15)  # within default tol of 1.0
    assert idx == 1
    np.testing.assert_allclose(ts.times("Omega"),  [0.0, 2.0])
    np.testing.assert_allclose(ts.values("Omega"), [0.0, 2.0])

def test_remove_point_no_match_raises_keyerror(base_abs):
    ts = base_abs
    with pytest.raises(KeyError):
        ts.remove_point("Omega", 1.1, tol=1e-6)  # nearest is 1.0 but outside tol

def test_remove_point_missing_channel_raises():
    ts = TimeSeries()
    with pytest.raises(KeyError):
        ts.remove_point("Omega", 0.0)


# ----------------------------- remove_at -----------------------------

def test_remove_at_middle(base_abs):
    ts = base_abs
    idx = ts.remove_at("Omega", 1)
    assert idx == 1
    np.testing.assert_allclose(ts.times("Omega"),  [0.0, 2.0])
    np.testing.assert_allclose(ts.values("Omega"), [0.0, 2.0])

def test_remove_at_first_and_last_leave_empty_channel():
    ts = TimeSeries()
    ts.add_series("Omega", [0.0], [5.0])
    # Remove index 0 (only sample)
    idx = ts.remove_at("Omega", 0)
    assert idx == 0
    # Channel still present but empty arrays (by current design)
    np.testing.assert_allclose(ts.times("Omega"),  np.array([]))
    np.testing.assert_allclose(ts.values("Omega"), np.array([]))

def test_remove_at_index_oob_raises(base_abs):
    ts = base_abs
    with pytest.raises(IndexError):
        ts.remove_at("Omega", -1)
    with pytest.raises(IndexError):
        ts.remove_at("Omega", 3)  # valid indices: 0..2

def test_remove_at_missing_channel_raises():
    ts = TimeSeries()
    with pytest.raises(KeyError):
        ts.remove_at("Omega", 0)


# ----------------------------- clear_channel & clear -----------------------------

def test_clear_channel_removes_key(base_norm):
    ts = base_norm
    assert ts.has_channel("Delta")
    ts.clear_channel("Delta")
    assert not ts.has_channel("Delta")
    # idempotent: clearing again should not raise
    ts.clear_channel("Delta")
    assert "Delta" not in ts.channels

def test_clear_removes_all_channels(base_norm):
    ts = base_norm
    assert len(ts) == 2
    ts.clear()
    assert len(ts) == 0
    assert ts.is_empty
    # domain() now has no channels -> should raise
    with pytest.raises(ValueError):
        ts.domain()

# ----------------------------- shift_time -----------------------------

def test_shift_time_no_channels_no_error():
    ts = TimeSeries()
    # Should be a no-op with no errors
    ts.shift_time(1.23)
    assert ts.is_empty


def test_shift_time_moves_all_channels_equally(ts_two_channels_abs):
    ts = ts_two_channels_abs
    # Original copies for comparison
    tO_before = ts.times("Omega").copy()
    tD_before = ts.times("Delta").copy()
    vO_before = ts.values("Omega").copy()
    vD_before = ts.values("Delta").copy()

    # Shift by +2.0 seconds
    ts.shift_time(2.0)

    # Times should all increase by dt; values unchanged
    np.testing.assert_allclose(ts.times("Omega"), tO_before + 2.0)
    np.testing.assert_allclose(ts.times("Delta"), tD_before + 2.0)
    np.testing.assert_allclose(ts.values("Omega"), vO_before)
    np.testing.assert_allclose(ts.values("Delta"), vD_before)

    # Strict monotonicity preserved (differences unchanged)
    np.testing.assert_allclose(np.diff(ts.times("Omega")), np.diff(tO_before))
    np.testing.assert_allclose(np.diff(ts.times("Delta")), np.diff(tD_before))

    # Domain shifts by +2.0
    dmin_before, dmax_before = 0.0, 2.0  # union domain of original fixture
    dmin_after, dmax_after = ts.domain()
    np.testing.assert_allclose(dmin_after, dmin_before + 2.0)
    np.testing.assert_allclose(dmax_after, dmax_before + 2.0)


def test_shift_time_negative_shift(ts_two_channels_abs):
    ts = ts_two_channels_abs
    ts.shift_time(-0.5)
    np.testing.assert_allclose(ts.times("Omega"), np.array([-0.5, 0.5, 1.5]))
    np.testing.assert_allclose(ts.times("Delta"), np.array([0.0, 1.5]))


# ----------------------------- normalize_start -----------------------------

def test_normalize_start_no_channels_is_noop():
    ts = TimeSeries()
    # Should not raise, and still empty
    ts.normalize_start()
    assert ts.is_empty


def test_normalize_start_moves_earliest_to_zero(ts_mixed_negative):
    ts = ts_mixed_negative
    # Earliest time across channels is -3.0 -> after normalize, add +3.0 to all
    ts.normalize_start()

    np.testing.assert_allclose(ts.times("Omega"), np.array([4.0, 5.0]))
    np.testing.assert_allclose(ts.times("Delta"), np.array([0.0, 2.0]))

    # Domain now starts at 0
    dmin, dmax = ts.domain()
    assert dmin == 0.0
    np.testing.assert_allclose(dmax, 5.0)


def test_normalize_preserves_relative_gaps(ts_two_channels_abs):
    ts = ts_two_channels_abs
    # Save gaps before
    gaps_O = np.diff(ts.times("Omega")).copy()
    gaps_D = np.diff(ts.times("Delta")).copy()

    ts.normalize_start()

    # Relative gaps must be identical after a constant shift
    np.testing.assert_allclose(np.diff(ts.times("Omega")), gaps_O)
    np.testing.assert_allclose(np.diff(ts.times("Delta")), gaps_D)

    # Earliest = 0
    dmin, _ = ts.domain()
    assert dmin == 0.0


def test_normalize_then_shift_roundtrip(ts_two_channels_abs):
    ts = ts_two_channels_abs
    # Compute intended shift to zero earliest time
    t0 = min(ts.times("Omega")[0], ts.times("Delta")[0])  # 0.0 for this fixture
    ts.normalize_start()
    # Shift back by -(-t0) == t0 (no-op for this fixture), assert equality to original
    ts.shift_time(t0)

    np.testing.assert_allclose(ts.times("Omega"), np.array([0.0, 1.0, 2.0]))
    np.testing.assert_allclose(ts.times("Delta"), np.array([0.5, 2.0]))

# ----------------------------- value_at: basic behavior -----------------------------

def test_value_at_interpolates_per_channel(ts_abs):
    # Omega: triangle 0->1->0 over [0,2]
    tq = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    out = ts_abs.value_at(tq, channels=["Omega"])
    y = out["Omega"]
    np.testing.assert_allclose(y, [0.0, 0.5, 1.0, 0.5, 0.0])

def test_value_at_zero_outside_domain(ts_abs):
    # Delta domain is [0.5, 1.5]
    tq = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    out = ts_abs.value_at(tq, channels=["Delta"])
    y = out["Delta"]
    # Inside domain use linear interpolation; outside -> 0
    # Between 0.5(10) and 1.5(12): at 1.0 expect 11.0
    np.testing.assert_allclose(y, [0.0, 10.0, 11.0, 12.0, 0.0])

def test_value_at_unknown_channel_returns_zeros(ts_abs):
    tq = [0.0, 0.5, 1.0]
    out = ts_abs.value_at(tq, channels=["NotAChannel"])
    np.testing.assert_array_equal(out["NotAChannel"], np.zeros(3, dtype=np.float64))

def test_value_at_all_present_when_channels_none(ts_abs):
    tq = [0.0, 1.0, 2.0]
    out = ts_abs.value_at(tq)  # all present channels
    assert set(out.keys()) == {"Omega", "Delta"}
    # spot-check shapes & types
    assert out["Omega"].shape == (3,)
    assert out["Delta"].dtype == np.float64

def test_value_at_empty_and_channels_none_returns_empty_dict():
    ts = TimeSeries()
    out = ts.value_at([0.0, 1.0, 2.0])
    assert out == {}

def test_value_at_boundaries_exact_values(ts_abs):
    # Exact sample points should give exact values
    out = ts_abs.value_at([0.0, 1.0, 2.0], channels=["Omega"])
    np.testing.assert_allclose(out["Omega"], [0.0, 1.0, 0.0])


# ----------------------------- value_at: input validation -----------------------------

def test_value_at_rejects_non_1d_input(ts_abs):
    with pytest.raises(ValueError):
        ts_abs.value_at([[0.0, 1.0]])  # 2D

def test_value_at_rejects_nan_inf(ts_abs):
    with pytest.raises(ValueError):
        ts_abs.value_at([0.0, np.nan, 1.0])
    with pytest.raises(ValueError):
        ts_abs.value_at([0.0, np.inf])

def test_value_at_accepts_generators(ts_abs):
    def gen():
        for t in [0.0, 0.5, 1.0]:
            yield t
    out = ts_abs.value_at(gen(), channels=["Omega"])
    np.testing.assert_allclose(out["Omega"], [0.0, 0.5, 1.0])


# ----------------------------- value_at_channel -----------------------------

def test_value_at_channel_single(ts_abs):
    y = ts_abs.value_at_channel("Omega", [0.0, 0.5, 1.0, 1.5, 2.0])
    np.testing.assert_allclose(y, [0.0, 0.5, 1.0, 0.5, 0.0])

def test_value_at_channel_unknown_returns_zeros(ts_abs):
    y = ts_abs.value_at_channel("Nope", [0.0, 0.5, 1.0])
    np.testing.assert_array_equal(y, np.zeros(3, dtype=np.float64))

def test_value_at_channel_accepts_generator(ts_abs):
    y = ts_abs.value_at_channel("Delta", (t for t in [0.0, 0.75, 1.5, 2.0]))
    np.testing.assert_allclose(y, [0.0, 10.5, 12.0, 0.0])

def test_value_at_channel_input_validation(ts_abs):
    with pytest.raises(ValueError):
        ts_abs.value_at_channel("Omega", [[0.0, 1.0]])  # 2D
    with pytest.raises(ValueError):
        ts_abs.value_at_channel("Omega", [0.0, np.nan])


# ----------------------------- normalized mode sanity -----------------------------

def test_value_at_normalized_mode_behaves_same_interpolation(ts_norm):
    # Interpolation/zeroing rules are identical in normalized mode
    yO = ts_norm.value_at_channel("Omega", [0.0, 0.25, 0.5, 0.75, 1.0, 1.5])
    yD = ts_norm.value_at_channel("Delta", [-0.5, 0.0, 0.5, 1.0, 1.5])
    np.testing.assert_allclose(yO, [0.0, 0.25, 0.5, 0.75, 1.0, 0.0])
    np.testing.assert_allclose(yD, [0.0, -1.0, 0.0, 1.0, 0.0])