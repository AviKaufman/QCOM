import time
import pytest
import qcom as qc

"""
This file is for testing the progress functions in qcom/progress.py using pytest.
To run the tests, use the command `pytest tests/test_progress.py` in the root directory of the repository.
If you get an error, do not push the changes to GitHub until the error is fixed.
"""


def test_progress_context(capsys):
    """
    Test that the progress context manager prints a starting message when entered,
    and a completed message when the context exits.
    """
    with qc.ProgressManager.progress("TestTask", total_steps=5):
        # Simulate some work.
        time.sleep(0.1)
    captured = capsys.readouterr().out
    assert "Starting: TestTask" in captured
    assert "Completed: TestTask" in captured


def test_update_progress(capsys):
    """
    Test that update_progress produces an output containing progress details.
    This test calls update_progress while within the progress context.
    """
    with qc.ProgressManager.progress("TestTask", total_steps=10):
        # Update progress at 50%
        qc.ProgressManager.update_progress(5)
        # Allow time for stdout to flush.
        time.sleep(0.05)
        out = capsys.readouterr().out
        # The update message should include the task name and 50.00% progress.
        assert "Task: TestTask" in out
        assert "50.00%" in out

    # Also check that the completion message is printed on context exit.
    captured = capsys.readouterr().out
    assert "Completed: TestTask" in captured


def test_dummy_context(capsys):
    """
    Test that the dummy context manager does nothing related to progress,
    so it does not print the progress messages.
    """
    with qc.ProgressManager.dummy_context():
        print("Inside dummy context")
    captured = capsys.readouterr().out
    # The output should only contain our print statement.
    assert "Inside dummy context" in captured
    assert "Starting:" not in captured
    assert "Completed:" not in captured


if __name__ == "__main__":
    # Test only this file
    pytest.main([__file__])
