# qcom/_internal/__init__.py
"""
QCOM Internal Utilities

This subpackage contains internal helper modules that are **not**
part of the stable public API. They are intended for internal use
within QCOM and may change without notice.

Modules
-------
- progress : The `ProgressManager` class for lightweight progress reporting.

Typical usage
-------------
Internal modules are used indirectly by higher-level components.
For example, progress tracking in `qcom.data.sampling`:

>>> from qcom._internal import ProgressManager
>>> with ProgressManager.progress("Example Task", total_steps=10):
...     for i in range(10):
...         ProgressManager.update_progress(i + 1)

Notes
-----
Users of QCOM should avoid depending directly on `_internal` unless
absolutely necessary. Prefer higher-level modules that wrap these utilities.
"""

from .progress import ProgressManager

__all__ = ["ProgressManager"]