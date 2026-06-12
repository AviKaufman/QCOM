from __future__ import annotations

from collections.abc import Callable

from qcom.controls.adapters.base import ControlAdapter
from qcom.controls.time_series import TimeSeries

_registry: dict[str, Callable[[TimeSeries], ControlAdapter]] = {}


def register_adapter(key: str, factory: Callable[[TimeSeries], ControlAdapter]) -> None:
    _registry[key.lower()] = factory


def get_adapter(key: str, series: TimeSeries) -> ControlAdapter:
    try:
        return _registry[key.lower()](series)
    except KeyError as exc:
        raise ValueError(f"No control adapter registered for key={key!r}") from exc
