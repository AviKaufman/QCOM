import numpy as np
import pytest
import time
from scipy.sparse import csr_matrix

# Import the qcom module as qc, which provides access to the utils functions.
import qcom as qc

"""
This file is for testing the utility functions in qcom/utils.py using pytest. 
To run the tests, use the command `pytest tests/test_utils` in the root directory of the repository.
If you get an error, do not push the changes to GitHub until the error is fixed.
"""


def test_find_eigenstate_numpy():
    """
    Test qc.find_eigenstate with a small symmetric numpy array.
    Using the 2x2 diagonal matrix:
        [[1, 0],
         [0, 2]]
    The smallest eigenvalue is 1 with eigenvector [1, 0] (up to sign).
    """
    h = np.array([[1, 0], [0, 2]], dtype=float)  # Use float dtype
    eigenvalue, eigenvector = qc.find_eigenstate(h, state_index=0, show_progress=False)
    # Check eigenvalue is approximately 1.
    assert np.isclose(eigenvalue, 1, atol=1e-6)
    # Eigenvector may be scaled or flipped in sign; compare absolute values.
    np.testing.assert_allclose(
        np.abs(eigenvector), np.array([1, 0], dtype=float), atol=1e-6
    )


def test_find_eigenstate_numpy_second_state():
    """
    Test qc.find_eigenstate with a numpy array when state_index is not zero.
    For the matrix:
        [[1, 0],
         [0, 2]]
    and state_index=1, we expect eigenvalue ~2 and eigenvector ~[0,1] (up to sign).
    """
    h = np.array([[1, 0], [0, 2]], dtype=float)  # Use float dtype
    eigenvalue, eigenvector = qc.find_eigenstate(h, state_index=1, show_progress=False)
    assert np.isclose(eigenvalue, 2, atol=1e-6)
    np.testing.assert_allclose(
        np.abs(eigenvector), np.array([0, 1], dtype=float), atol=1e-6
    )


def test_find_eigenstate_sparse():
    """
    Test qc.find_eigenstate with a sparse matrix.
    The function should convert the sparse matrix to a numpy array.
    Using the matrix:
        [[3, 0],
         [0, 4]]
    The smallest eigenvalue is 3.
    """
    h = np.array([[3, 0], [0, 4]], dtype=float)  # Use float dtype
    sparse_h = csr_matrix(h)
    eigenvalue, eigenvector = qc.find_eigenstate(
        sparse_h, state_index=0, show_progress=False
    )
    assert np.isclose(eigenvalue, 3, atol=1e-6)
    np.testing.assert_allclose(
        np.abs(eigenvector), np.array([1, 0], dtype=float), atol=1e-6
    )


def test_find_eigenstate_with_progress(capsys):
    """
    Test qc.find_eigenstate with show_progress=True.
    This captures stdout to check that progress messages are printed.
    """
    h = np.array([[1, 0], [0, 2]], dtype=float)  # Use float dtype
    eigenvalue, eigenvector = qc.find_eigenstate(h, state_index=0, show_progress=True)
    # Capture the output printed during progress.
    captured = capsys.readouterr().out
    # Check that progress messages appear (either the starting or finishing message).
    assert "Finding Eigenstate" in captured or "Eigenstate" in captured
    assert np.isclose(eigenvalue, 1, atol=1e-6)


def test_order_dict():
    """
    Test qc.order_dict by providing a dictionary with binary string keys.
    The keys are interpreted as integers.
    Example:
        {"10": 1, "01": 2, "11": 3, "00": 4}
    When sorted by int(key, 2), the order should be:
        "00" (0), "01" (1), "10" (2), "11" (3)
    """
    input_dict = {"10": 1, "01": 2, "11": 3, "00": 4}
    expected_keys = ["00", "01", "10", "11"]
    ordered = qc.order_dict(input_dict)
    assert list(ordered.keys()) == expected_keys


def test_part_dict():
    """
    Test qc.part_dict by extracting specific bit positions.
    Given:
        input_dict = {"101": 1, "111": 2, "000": 3, "010": 4}
    and indices = [0, 2]:
      - "101" -> bits at indices 0 and 2 are "1" and "1" => "11"
      - "111" -> "11"
      - "000" -> "00"
      - "010" -> "00"
    So the expected output is:
        {"11": 1 + 2, "00": 3 + 4} -> {"11": 3, "00": 7}
    """
    input_dict = {"101": 1, "111": 2, "000": 3, "010": 4}
    expected_output = {"11": 3, "00": 7}
    result = qc.part_dict(input_dict, [0, 2])
    assert result == expected_output


def test_part_dict_empty_indices():
    """
    Test qc.part_dict with an empty list of indices.
    In this case, every key will be mapped to an empty string, so all values should sum together.
    """
    input_dict = {"101": 1, "111": 2, "000": 3, "010": 4}
    expected_output = {"": 1 + 2 + 3 + 4}  # All keys become "".
    result = qc.part_dict(input_dict, [])
    assert result == expected_output


if __name__ == "__main__":
    # run tests for only this file
    pytest.main([__file__])
