from __future__ import annotations

import pytest

from qcom.controls.adapters import get_adapter, register_adapter
from qcom.controls.time_series import TimeSeries


class _DummyAdapter:
    def __init__(self, series: TimeSeries) -> None:
        self.series = series

    @property
    def required_channels(self) -> tuple[str, ...]:
        return ("Omega",)

    def hamiltonian_at(self, t: float, controls: dict[str, float]) -> object:
        return {"t": t, "controls": dict(controls)}


def test_register_adapter_is_case_insensitive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("qcom.controls.adapters._registry", {})
    register_adapter("RyDBeRg", _DummyAdapter)

    series = TimeSeries(mode="absolute", Omega=([0.0], [1.0]))
    adapter = get_adapter("rydberg", series)

    assert isinstance(adapter, _DummyAdapter)
    assert adapter.series is series


def test_get_adapter_raises_for_unknown_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("qcom.controls.adapters._registry", {})

    series = TimeSeries(mode="absolute", Omega=([0.0], [1.0]))
    with pytest.raises(ValueError, match="No control adapter registered"):
        get_adapter("missing", series)
