import time
import sys
from contextlib import contextmanager

"""
Progress Manager for QCOM Project. Helps to track progress of long running tasks.
"""


class ProgressManager:
    """Manages progress updates for long-running tasks with real-time feedback."""

    active_task = None  # Keeps track of the currently active task
    total_steps = None  # Total number of steps for the current task
    start_time = None  # Start time of the current task

    @staticmethod
    @contextmanager
    def progress(task_name, total_steps=None):
        if ProgressManager.active_task is not None:
            # If there's already an active task, skip nested progress tracking
            yield
            return

        # Initialize progress tracking
        ProgressManager.active_task = task_name
        ProgressManager.total_steps = total_steps
        ProgressManager.start_time = time.time()
        sys.stdout.write(f"Starting: {task_name}...\n")
        sys.stdout.flush()

        try:
            yield
        finally:
            # Compute total elapsed time
            elapsed_time = time.time() - ProgressManager.start_time
            sys.stdout.write(
                f"\nCompleted: {task_name}. Elapsed time: {elapsed_time:.2f} seconds.\n"
            )
            sys.stdout.flush()

            # Reset state
            ProgressManager.active_task = None
            ProgressManager.total_steps = None
            ProgressManager.start_time = None

    @staticmethod
    def update_progress(current_step):
        if ProgressManager.active_task is None or ProgressManager.total_steps is None:
            return

        elapsed_time = time.time() - ProgressManager.start_time
        percent_finished = (current_step / ProgressManager.total_steps) * 100
        estimated_total_time = elapsed_time / (
            current_step / ProgressManager.total_steps
        )
        estimated_remaining_time = estimated_total_time - elapsed_time

        message = (
            f"Task: {ProgressManager.active_task} | "
            f"Progress: {percent_finished:.2f}% | "
            f"Elapsed: {elapsed_time:.2f}s | "
            f"Remaining: {estimated_remaining_time:.2f}s"
        )
        # Clear the line before writing the new message
        sys.stdout.write("\r" + " " * 80 + "\r")
        sys.stdout.write(message)
        sys.stdout.flush()

    @staticmethod
    @contextmanager
    def dummy_context():
        """No-op context manager for when progress tracking is disabled."""
        yield
