"""
QCOM Controls

This subpackage provides control primitives for time-dependent simulations,
such as representing classical drive fields or detuning profiles applied
to quantum systems.

Modules
-------
- time_series : The `TimeSeries` class for representing piecewise-constant,
                piecewise-linear, or general sampled control functions.

Typical usage
-------------
>>> from qcom.controls import TimeSeries
>>> ts = TimeSeries(Omega=([0.0, 1.0, 2.0], [0.0, 0.5, 1.0]))
>>> ts.value_at_channel("Omega", [0.5])[0]
0.25

Notes
-----
Currently, this namespace exposes only `TimeSeries`. In the future, it may
grow to include additional control constructs (e.g. pulses, ramps, composite
sequences).
"""

from .time_series import TimeSeries

__all__ = ["TimeSeries"]
