# qcom/_internal/progress.py
"""
Progress utilities for long-running tasks in QCOM.

Goal
----
Provide a tiny, zero-dependency progress manager that offers nested-friendly,
stdout-based status reporting for notebooks and scripts.

Scope (current)
---------------
• Context-managed progress scopes:
    with ProgressManager.progress("Task", total_steps=N): ...
• Live single-line updates with % complete, ETA, and elapsed time.
• Safe nested usage: only the outermost scope prints; inner scopes are no-ops.
• Guarded against division-by-zero and mis-nesting.

Design notes
------------
• Minimal footprint: no external tqdm dependency.
• Clean API: progress(), update_progress(), dummy_context().
• Output is human-readable and stable across terminals.

Typical usage
-------------
>>> from qcom._internal.progress import ProgressManager
>>> with ProgressManager.progress("Heavy thing", total_steps=100):
...     for i in range(100):
...         # ... work ...
...         ProgressManager.update_progress(i + 1)

API
---
- ProgressManager.progress(task_name, total_steps=None)
    Context manager that begins/ends a progress scope.
- ProgressManager.update_progress(current_step)
    Update the current step (clamped to [0, total_steps]).
- ProgressManager.dummy_context()
    No-op context manager for conditional progress paths.

Future extensions (non-breaking)
--------------------------------
• Optional log-level routing (stdout vs. logging).
• Pluggable renderers (ASCII bars, rich, Jupyter widgets).
• Rate smoothing and adaptive ETA.
"""

# ------------------------------------------ Imports ------------------------------------------

import time
import sys
from contextlib import contextmanager


# ========================================== Progress Manager ==========================================

class ProgressManager:
    """
    Manage progress updates for long-running tasks with real-time feedback.

    Nested scopes are supported; only the *outermost* active scope emits output.
    """

    # ------------------------------------------ Internal State ------------------------------------------
    # Stack of active scopes; each item is a dict(task, total, start, last_msg_len)
    _stack = []

    # ------------------------------------------ Public API: Context Scope ------------------------------------------

    @staticmethod
    @contextmanager
    def progress(task_name, total_steps=None):
        """
        Begin a progress scope.

        Args:
            task_name (str): Human-readable name for the task (shown in output).
            total_steps (int | None): If provided, enables % complete and ETA reporting.
                                      If None, prints only 'Starting'/'Completed'.

        Notes:
            Only the *outermost* active scope prints live updates; nested scopes
            run silently to avoid stdout spam.
        """
        frame = {
            "task": str(task_name),
            "total": int(total_steps) if total_steps is not None else None,
            "start": time.time(),
            "last_msg_len": 0,
        }

        # Determine if this is the top-level (printing) frame
        is_top = (len(ProgressManager._stack) == 0)
        ProgressManager._stack.append(frame)

        if is_top:
            sys.stdout.write(f"Starting: {frame['task']}...\n")
            sys.stdout.flush()

        try:
            yield
        finally:
            # Pop exactly this frame (defensive against mis-nesting)
            if ProgressManager._stack and ProgressManager._stack[-1] is frame:
                ProgressManager._stack.pop()
            else:
                # If nesting got weird, clear the whole stack defensively
                ProgressManager._stack = []

            if is_top:
                elapsed = time.time() - frame["start"]
                # Finish the progress line cleanly
                sys.stdout.write("\r" + " " * 80 + "\r")
                sys.stdout.write(
                    f"Completed: {frame['task']}. Elapsed time: {elapsed:.2f} seconds.\n"
                )
                sys.stdout.flush()

    # ------------------------------------------ Public API: Update ------------------------------------------

    @staticmethod
    def update_progress(current_step: int):
        """
        Update the progress display for the active (outermost) task.

        Args:
            current_step (int): Current step number (clamped into [0, total_steps]).

        Notes:
            • If no scope is active or `total_steps` was None, this is a no-op.
            • ETA is shown only when `current_step > 0`.
        """
        if not ProgressManager._stack:
            return  # no active task

        frame = ProgressManager._stack[-1]
        total = frame["total"]
        if total is None or total <= 0:
            return  # caller didn't specify a total → nothing to format

        # Clamp and avoid zero-division
        cur = max(0, min(int(current_step), total))
        elapsed = time.time() - frame["start"]

        pct = (cur / total) * 100.0
        if cur > 0:
            rate = cur / max(elapsed, 1e-12)
            remaining = (total - cur) / rate
        else:
            remaining = float("inf")

        msg = (
            f"Task: {frame['task']} | "
            f"Progress: {pct:6.2f}% ({cur}/{total}) | "
            f"Elapsed: {elapsed:7.2f}s | "
            f"Remaining: {'∞' if not (remaining < 1e19) else f'{remaining:7.2f}s'}"
        )

        # Clear previous line then write the new message
        sys.stdout.write("\r" + " " * 80 + "\r")
        sys.stdout.write(msg)
        sys.stdout.flush()

    # ------------------------------------------ Public API: No-op Context ------------------------------------------

    @staticmethod
    @contextmanager
    def dummy_context():
        """
        No-op context manager for when progress tracking is disabled.

        Usage:
            with (ProgressManager.progress("Task", N) if show_progress else ProgressManager.dummy_context()):
                ...
        """
        yield