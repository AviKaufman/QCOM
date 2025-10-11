# qcom/io/__init__.py
"""
qcom.io
=======

Input/output utilities for experiment and simulation data.

Design
------
- Provides a unified interface for loading/saving state–probability dictionaries
  from multiple file formats (JSON, Parquet, text).
- All loaders follow the convention:
      parse_<format>(...) -> (data_dict, total_count) or dict
- All savers follow the convention:
      save_<format>(...) -> None
- Progress reporting is integrated via ProgressManager.

Typical usage
-------------
from qcom.io import parse_json, parse_parquet, parse_file
from qcom.io import save_dict_to_parquet, save_data

# Parse data
data, total = parse_json("aquila_output.json")
data = parse_parquet("results.parquet")
data, total = parse_file("results.txt")

# Save data
save_dict_to_parquet(data, "results.parquet")
save_data(data, "results.txt")

Public API
----------
- parse_json, parse_parquet, parse_file
- save_dict_to_parquet, save_data
"""

from __future__ import annotations
from typing import TYPE_CHECKING

__all__ = [
    "parse_json",
    "parse_parquet",
    "parse_file",
    "save_dict_to_parquet",
    "save_data",
]

# Type-only imports (no runtime cost)
if TYPE_CHECKING:
    from typing import Dict, Tuple


# --- Lazy attribute loader ----------------------------------------------------
def __getattr__(name: str):
    if name in {"parse_json"}:
        from . import aquila as _aquila
        return getattr(_aquila, name)

    if name in {"parse_parquet", "save_dict_to_parquet"}:
        from . import parquet as _parquet
        return getattr(_parquet, name)

    if name in {"parse_file", "save_data"}:
        from . import text as _text
        return getattr(_text, name)

    raise AttributeError(f"module 'qcom.io' has no attribute {name!r}")


def __dir__():
    # Helps IDEs / tab-completion see the lazily exposed names
    return sorted(list(globals().keys()) + __all__)