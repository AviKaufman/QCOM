# qcom/controls/adapters/__init__.py
from __future__ import annotations
from typing import Type, Callable, Dict
from qcom.controls.time_series import TimeSeries
from qcom.controls.adapters.base import ControlAdapter

_registry: Dict[str, Callable[[TimeSeries], ControlAdapter]] = {}

def register_adapter(key: str, factory: Callable[[TimeSeries], ControlAdapter]) -> None:
    _registry[key.lower()] = factory

def get_adapter(key: str, series: TimeSeries) -> ControlAdapter:
    try:
        return _registry[key.lower()](series)
    except KeyError:
        raise ValueError(f"No control adapter registered for key={key!r}")

# Register Rydberg
from .rydberg import RydbergAdapter
register_adapter("rydberg", lambda ts: RydbergAdapter(ts))