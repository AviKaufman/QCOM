import numpy as np
import pytest
import qcom as qc

"""
This file is for testing the classical information functions in qcom/classical_info.py using pytest.
To run the tests, use the command `pytest tests/test_classical_info.py` in the root directory of the repository.
If you get an error, do not push the changes to GitHub until the error is fixed.
"""


def test_shannon_entropy_uniform():
    """
    For a uniform distribution over 4 states, each with probability 0.25,
    the Shannon entropy is given by:
      - sum(0.25 * ln(0.25)) * 4 = - ln(0.25) = ln(4)
    """
    prob_dict = {"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}
    expected = np.log(4)
    result = qc.compute_shannon_entropy(prob_dict)
    np.testing.assert_allclose(result, expected, atol=1e-10)


def test_shannon_entropy_degenerate():
    """
    For a degenerate distribution where one state has all the probability,
    the Shannon entropy is 0.
    """
    prob_dict = {"a": 1.0, "b": 0.0, "c": 0.0}
    expected = 0.0
    result = qc.compute_shannon_entropy(prob_dict)
    np.testing.assert_allclose(result, expected, atol=1e-10)


def test_shannon_entropy_non_normalized():
    """
    For a non-normalized distribution, e.g., {'a': 2, 'b': 2, 'c': 6},
    the probabilities are normalized as:
      0.2, 0.2, and 0.6.
    The Shannon entropy is:
      -(0.2*ln(0.2) + 0.2*ln(0.2) + 0.6*ln(0.6))
    """
    prob_dict = {"a": 2, "b": 2, "c": 6}
    expected = -(0.2 * np.log(0.2) + 0.2 * np.log(0.2) + 0.6 * np.log(0.6))
    result = qc.compute_shannon_entropy(prob_dict)
    np.testing.assert_allclose(result, expected, atol=1e-10)


def test_shannon_entropy_custom():
    """
    For a custom probability distribution, e.g., {'a': 0.5, 'b': 0.3, 'c': 0.2},
    the Shannon entropy is:
      -(0.5*ln(0.5) + 0.3*ln(0.3) + 0.2*ln(0.2))
    """
    prob_dict = {"a": 0.5, "b": 0.3, "c": 0.2}
    expected = -(0.5 * np.log(0.5) + 0.3 * np.log(0.3) + 0.2 * np.log(0.2))
    result = qc.compute_shannon_entropy(prob_dict)
    np.testing.assert_allclose(result, expected, atol=1e-10)


if __name__ == "__main__":
    # Run the tests from only this file
    pytest.main([__file__])
