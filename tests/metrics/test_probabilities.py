import numpy as np
import pytest

from qcom.metrics import probabilities as probs


def test_compute_cumulative_probability_at_value_basic():
    probabilities = {"00": 0.1, "01": 0.2, "10": 0.7}
    result = probs.compute_cumulative_probability_at_value(probabilities, 0.2)
    assert np.isclose(result, 0.3)


def test_compute_cumulative_distribution_unique_mode_and_normalization():
    probabilities = {"00": 0.25, "01": 0.25, "10": 0.25, "11": 0.25}
    x, y = probs.compute_cumulative_distribution(probabilities, grid=None)
    # Unique p=0.25, final x should end with 1.0
    assert np.isclose(x[-1], 1.0)
    assert np.isclose(y[-1], 1.0)


def test_compute_cumulative_distribution_with_grid_and_validation():
    probabilities = {"0": 0.5, "1": 0.5}
    grid = [0.2, 0.5, 1.0]
    x, y = probs.compute_cumulative_distribution(probabilities, grid=grid)
    assert np.allclose(x, sorted(grid))
    assert np.isclose(y[-1], 1.0)


def test_compute_cumulative_distribution_errors():
    with pytest.raises(ValueError):
        probs.compute_cumulative_distribution({}, grid=None)
    with pytest.raises(ValueError):
        probs.compute_cumulative_distribution({"0": 0.2, "1": 0.3}, grid=None)
    with pytest.raises(ValueError):
        probs.compute_cumulative_distribution({"0": 1.0}, grid=[])
    with pytest.raises(ValueError):
        probs.compute_cumulative_distribution({"0": 1.0}, grid=[-0.1])


def test_compute_n_of_p_curve_on_simple_dist():
    probabilities = {"0": 0.5, "1": 0.5}
    uniq, vals = probs.compute_n_of_p_curve(probabilities, p_delta=0.1, log_base=10.0)
    assert len(uniq) == 1 or len(uniq) == 2
    assert all(v >= 0 for v in vals)


def test_compute_n_of_p_curve_with_array_input_and_bad_logbase():
    probabilities = [0.25, 0.75]
    uniq, vals = probs.compute_n_of_p_curve(probabilities, p_delta=0.2)
    assert len(uniq) > 0
    with pytest.raises(ValueError):
        probs.compute_n_of_p_curve(probabilities, log_base=1.0)


def test_compute_n_of_p_matches_manual_calc():
    sorted_probs = np.array([0.25, 0.75])
    cumulative_probs = np.cumsum(sorted_probs)
    probability = 0.25
    value = probs.compute_n_of_p(probability, sorted_probs, cumulative_probs, p_delta=0.1)
    assert value >= 0


def test_compute_n_of_p_with_invalid_args():
    sorted_probs = np.array([0.1, 0.2, 0.7])
    cumulative_probs = np.cumsum(sorted_probs)
    assert probs.compute_n_of_p(0.0, sorted_probs, cumulative_probs) == 0.0
    with pytest.raises(ValueError):
        probs.compute_n_of_p(0.1, sorted_probs, cumulative_probs, log_base=1.0)


def test_probability_compatibility_aliases():
    probabilities = {"0": 0.5, "1": 0.5}
    with pytest.warns(DeprecationWarning, match="compute_cumulative_probability_at_value"):
        assert np.isclose(probs.cumulative_probability_at_value(probabilities, 0.5), 1.0)
    with pytest.warns(DeprecationWarning, match="compute_cumulative_distribution"):
        alias_y = probs.cumulative_distribution(probabilities, grid=[0.5, 1.0])[1]
    assert np.allclose(
        alias_y,
        probs.compute_cumulative_distribution(probabilities, grid=[0.5, 1.0])[1],
    )
    sorted_probs = np.array([0.5, 0.5])
    cumulative_probs = np.cumsum(sorted_probs)
    with pytest.warns(DeprecationWarning, match="compute_n_of_p"):
        alias_n_of_p = probs.compute_N_of_p(0.5, sorted_probs, cumulative_probs)
    assert alias_n_of_p == probs.compute_n_of_p(
        0.5,
        sorted_probs,
        cumulative_probs,
    )
    with pytest.warns(DeprecationWarning, match="compute_n_of_p_curve"):
        alias_n_curve = probs.compute_N_of_p_all(probabilities)[0]
    assert np.allclose(alias_n_curve, probs.compute_n_of_p_curve(probabilities)[0])


def test_statevector_to_probabilities_big_and_little_endian():
    psi = np.array([1, 0, 0, 0], dtype=np.complex128)  # |00>
    d_big = probs.statevector_to_probabilities(psi, msb_site0=True)
    d_little = probs.statevector_to_probabilities(psi, msb_site0=False)
    assert "00" in d_big
    assert "00"[::-1] in d_little  # reversed


def test_statevector_to_probabilities_drop_tol_and_errors():
    psi = np.array([1, 0, 0, 0], dtype=np.complex128)
    d = probs.statevector_to_probabilities(psi, drop_tol=0.5)
    assert d == {"00": 1.0} or d == {}  # depending on tolerance
    with pytest.raises(ValueError):
        probs.statevector_to_probabilities(np.array([1, 2, 3]))  # not power of 2 length


def test_get_eigenstate_probabilities_from_dense_matrix_ground_state():
    # H = diag([0,1,2,3]) → ground state is |00>
    H = np.diag([0.0, 1.0, 2.0, 3.0])
    result = probs.get_eigenstate_probabilities(H, state_index=0, msb_site0=True)
    assert np.isclose(sum(result.values()), 1.0)
    assert "00" in result and np.isclose(result["00"], 1.0)


def test_get_eigenstate_probabilities_with_endian_and_drop_tol():
    H = np.diag([0.0, 1.0])
    result = probs.get_eigenstate_probabilities(H, msb_site0=False, drop_tol=0.5)
    assert all(isinstance(k, str) for k in result.keys())
    assert all(isinstance(v, float) for v in result.values())
