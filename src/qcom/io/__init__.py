"""
qcom.io
=======

Input/output utilities for experiment and simulation data.

Design
------
- Provides a unified interface for loading/saving state–probability dictionaries
  from multiple file formats (JSON, Parquet, text).
- Progress reporting is integrated via ProgressManager.

Typical usage
-------------
from qcom.io import parse_aquila_json, parse_parquet, parse_text
from qcom.io import save_parquet, save_text

# Parse data
data, total = parse_aquila_json("aquila_output.json")
data = parse_parquet("results.parquet")
data, total = parse_text("results.txt")

# Save data
save_parquet(data, "results.parquet")
save_text(data, "results.txt")

Public API
----------
- Preferred: parse_aquila_json, parse_parquet, parse_text, save_parquet, save_text
- Deprecated compatibility aliases: parse_json, parse_file, save_dict_to_parquet, save_data
"""

from __future__ import annotations

__all__ = [
    "parse_aquila_json",
    "parse_json",
    "parse_parquet",
    "parse_text",
    "parse_file",
    "save_parquet",
    "save_dict_to_parquet",
    "save_text",
    "save_data",
]


def __getattr__(name: str):
    if name in {"parse_aquila_json", "parse_json"}:
        from . import aquila as _aquila

        return getattr(_aquila, name)

    if name in {"parse_parquet", "save_parquet", "save_dict_to_parquet"}:
        from . import parquet as _parquet

        return getattr(_parquet, name)

    if name in {"parse_text", "parse_file", "save_text", "save_data"}:
        from . import text as _text

        return getattr(_text, name)

    raise AttributeError(f"module 'qcom.io' has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + __all__)
