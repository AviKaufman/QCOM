"""Internal helpers for compatibility deprecation warnings."""

from __future__ import annotations

import warnings


def warn_deprecated_alias(old_name: str, new_name: str) -> None:
    """Warn that a compatibility alias should be replaced by its preferred name."""
    warnings.warn(
        f"`{old_name}` is deprecated and will be removed in a future release; "
        f"use `{new_name}` instead.",
        DeprecationWarning,
        stacklevel=3,
    )
