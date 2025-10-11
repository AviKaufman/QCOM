"""
TimeSeries — container for time-dependent control envelopes

Goal
----
Provide a transparent, NumPy-centric container for time-dependent control
signals used to evolve QCOM Hamiltonians. The class enforces a clear,
minimal contract: a set of named channels (e.g. "Omega", "Delta", "Phi"),
each defined by an explicit (t, value) schedule. The object is responsible
only for interpolation and storage; it does not embed site-dependent scaling.

Key guarantees
--------------
• Channels are independent: each channel has its own strictly increasing
  time axis and values. The global domain is the union of all channel domains.
• Storage is float64 NumPy arrays. Insertion keeps arrays strictly sorted.
• Interpolation is always linear between sample points.
• Queries outside a channel's domain (before first or after last sample)
  return 0.0 for that channel.
• Unspecified channels default to a constant 0.0 envelope.
• Duplicate times within a channel are disallowed; if a new sample is
  inserted at an existing time (within tolerance), the value is replaced.

Design philosophy
-----------------
• Pure container: TimeSeries does not carry atom- or site-level parameters.
  It only provides envelopes. Site dependence (e.g., Ω_max[i], Δ_span[i])
  is applied later at evolve time. This separation avoids hidden coupling
  and keeps the class reusable across experiments.

• Absolute vs. normalized envelopes:
  - Absolute mode: values are physical quantities (e.g. Ω in MHz). This is
    the correct mode when all sites share identical parameters.
  - Normalized envelope mode: values are dimensionless
      - Ω_env ∈ [0, 1]
      - Δ_env ∈ [-1, 1]
      - Φ_env unbounded
    At evolve time, users scale these by per-site maxima or spans to produce
    site-dependent controls. This avoids duplicating per-site series inside
    the container and enforces clarity: use absolute mode for globally uniform
    systems, normalized mode for site-dependent systems.

Editing & mutability
--------------------
• Users may build schedules incrementally:
  - 'add_point(name, t, value)' inserts a single sample at the correct place.
  - 'add_series(name, times, values)' merges multiple samples in order.
• Samples are automatically sorted by time.
• Duplicate times are replaced rather than duplicated.
• Deletion and clearing are supported:
  - 'remove_point', 'remove_at', 'clear_channel', 'clear'.

Primary use cases
-----------------
• Author arbitrary time series manually or from experimental logs.
• Incrementally build or edit schedules with add/remove operations.
• Interpolate channel values at arbitrary query times during simulation.
• Provide envelopes for Hamiltonian evolution, with site dependence applied
  externally at evolve time.

Future extensions (non-breaking)
--------------------------------
• Support for alternate interpolation schemes (e.g., spline).
• Convenience constructors (e.g., ramps, sinusoidal templates).
• Import/export formats (JSON, CSV, experiment logs).
"""

import numpy as np
from typing import Literal, Mapping
from types import MappingProxyType
from collections.abc import Iterable

# ------------------------------------------ TimeSeries Class ------------------------------------------

class TimeSeries:
    """
    Container for time-dependent control envelopes.

    See file-level docstring for design guarantees.
    """

    # ------------------------------------------ Constructor and Attributes ------------------------------------------

    def __init__(
        self,
        mode: Literal["absolute", "normalized"] = "absolute",
        **channels: tuple[Iterable[float], Iterable[float]],
    ):
        """
        Initialize a TimeSeries.

        Args:
            mode ({"absolute","normalized"}, default="absolute"):
                Interpretation of values:
                  - "absolute": physical units (e.g., rad/s).
                  - "normalized": Ω ∈ [0, 1], Δ ∈ [-1, 1], Φ unbounded.
            **channels:
                Optional initial schedules as keyword args: name=(times, values),
                where both are 1D, same length, finite, strictly increasing times.
                Times may be any real numbers (negative allowed); only strict monotonicity is required.
        """
        if mode not in ("absolute", "normalized"):
            raise ValueError("mode must be 'absolute' or 'normalized'")
        self.mode = mode

        # Internal storage: channel -> (times, values), both float64 contiguous
        self._channels: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        for name, (times, values) in channels.items():
            # Coerce
            t = np.asarray(times, dtype=np.float64)
            v = np.asarray(values, dtype=np.float64)

            # Shape & monotonic checks
            if t.ndim != 1 or v.ndim != 1 or t.shape[0] != v.shape[0]:
                raise ValueError(f"Channel '{name}' must have 1D times/values of equal length")
            if t.size == 0:
                raise ValueError(f"Channel '{name}' must contain at least one sample")
            if not np.isfinite(t).all():
                raise ValueError(f"Channel '{name}' times contain NaN/Inf")
            if not np.isfinite(v).all():
                raise ValueError(f"Channel '{name}' values contain NaN/Inf")
            if not np.all(np.diff(t) > 0.0):
                raise ValueError(f"Channel '{name}' times must be strictly increasing")

            # Normalized-mode bounds
            if self.mode == "normalized":
                lname = name.lower()
                if lname == "omega":
                    if (v < 0.0).any() or (v > 1.0).any():
                        vmin, vmax = float(v.min()), float(v.max())
                        raise ValueError(f"Normalized Omega must be in [0,1]; got min={vmin}, max={vmax}")
                elif lname == "delta":
                    if (v < -1.0).any() or (v > 1.0).any():
                        vmin, vmax = float(v.min()), float(v.max())
                        raise ValueError(f"Normalized Delta must be in [-1,1]; got min={vmin}, max={vmax}")
                # Phi (phase) intentionally unbounded

            # Normalize storage
            t = np.ascontiguousarray(t, dtype=np.float64)
            v = np.ascontiguousarray(v, dtype=np.float64)
            self._channels[name] = (t, v)


    # ------------------------------------------ Properties ------------------------------------------
    
    @property
    def channels(self) -> Mapping[str, tuple[np.ndarray, np.ndarray]]:
        """
        Read-only snapshot of all channels.

        Returns:
            Mapping[str, (times, values)] with **read-only** NumPy views.
        """
        snap: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name, (t, v) in self._channels.items():
            t_view = t.view(); t_view.setflags(write=False)
            v_view = v.view(); v_view.setflags(write=False)
            snap[name] = (t_view, v_view)
        return MappingProxyType(snap)

    # ------------------------------------------ New Property ------------------------------------------

    @property
    def channel_names(self) -> tuple[str, ...]:
        return tuple(self._channels.keys())
    
    # ------------------------------------------ New Property ------------------------------------------

    @property
    def is_empty(self) -> bool:
        return not self._channels


    # ------------------------------ Read-only accessors ------------------------------

    def __len__(self) -> int:
        return len(self._channels)
    
    # ------------------------------ New Read-only ------------------------------

    def has_channel(self, name: str) -> bool:
        """
        Check whether a channel exists.

        Args:
            name: Channel name (case-sensitive).

        Returns:
            bool: True if present.
        """
        return name in self._channels
    
    # ------------------------------ New Read-only ------------------------------

    def times(self, name: str) -> np.ndarray:
        """
        Read-only view of the time grid for a channel.

        Args:
            name: Channel name.

        Returns:
            np.ndarray: 1D float64 array (strictly increasing).

        Raises:
            KeyError: If the channel does not exist.
        """
        try:
            t = self._channels[name][0]
        except KeyError:
            raise KeyError(f"Unknown channel '{name}'") from None
        view = t.view()
        view.setflags(write=False)
        return view

    # ------------------------------ New Read-only ------------------------------

    def values(self, name: str) -> np.ndarray:
        """
        Read-only view of the values for a channel.

        Args:
            name: Channel name.

        Returns:
            np.ndarray: 1D float64 array aligned with 'times(name)'.

        Raises:
            KeyError: If the channel does not exist.
        """
        try:
            v = self._channels[name][1]
        except KeyError:
            raise KeyError(f"Unknown channel '{name}'") from None
        view = v.view()
        view.setflags(write=False)
        return view

    # ------------------------------ New Read-only ------------------------------

    def domain(self, name: str | None = None) -> tuple[float, float]:
        """
        Return the time domain.

        Args:
            name: If provided, return domain for that channel only.
                  If None, return the **union** domain across all channels.

        Returns:
            (t_min, t_max): Floats delimiting the domain.

        Raises:
            KeyError: If 'name' is provided and the channel does not exist.
            ValueError: If there are no channels (union domain undefined).
        """
        if name is not None:
            try:
                t = self._channels[name][0]
            except KeyError:
                raise KeyError(f"Unknown channel '{name}'") from None
            return float(t[0]), float(t[-1])

        # Union domain across all channels
        if not self._channels:
            raise ValueError("domain(): no channels present (empty TimeSeries).")
        tmins = [ch[0][0] for ch in self._channels.values()]
        tmaxs = [ch[0][-1] for ch in self._channels.values()]
        return float(min(tmins)), float(max(tmaxs))

    # ------------------------------ Representation ------------------------------

    def __repr__(self) -> str:
        """
        Concise summary string for debugging/inspection.
        """
        parts = []
        for name, (t, _) in self._channels.items():
            parts.append(f"{name}[L={t.size}, {t[0]:.3e}→{t[-1]:.3e}s]")
        body = ", ".join(parts) if parts else "∅"
        mode = self.mode
        return f"TimeSeries(mode={mode!r}, channels={body})"

    # ------------------------------------------ Utilities ------------------------------------------

    @staticmethod
    def _rel_tol(t: float) -> float:
        """
        Relative time tolerance used to detect 'the same sample time'.
        """
        return 1e-12 * max(1.0, abs(float(t)))

    # ------------------------------------------ New Utility ------------------------------------------

    def _check_normalized_bounds(self, name: str, values: np.ndarray) -> None:
        """
        Enforce normalized-mode bounds for well-known channels.
        (No-op in absolute mode.)
        """
        if self.mode != "normalized":
            return
        lname = name.lower()
        if lname == "omega":
            if (values < 0.0).any() or (values > 1.0).any():
                vmin, vmax = float(values.min()), float(values.max())
                raise ValueError(f"Normalized Omega must be in [0,1]; got min={vmin}, max={vmax}")
        elif lname == "delta":
            if (values < -1.0).any() or (values > 1.0).any():
                vmin, vmax = float(values.min()), float(values.max())
                raise ValueError(f"Normalized Delta must be in [-1,1]; got min={vmin}, max={vmax}")
        # Phi unbounded by design

    # ------------------------------------------ Modifiers ------------------------------------------

    def add_point(self, name: str, t: float, value: float, *, tol: float | None = None) -> int:
        """
        Insert or replace a single sample (t, value) into channel 'name'.

        Behavior:
            • Keeps the channel's time array strictly increasing.
            • If 't' matches an existing sample within 'tol' (relative default),
              the existing value is REPLACED (no duplicate time created).
            • If channel does not exist, it is created.

        Args:
            name: Channel name (case-sensitive).
            t: Sample time (seconds).
            value: Sample value (units depend on mode).
            tol: Absolute tolerance for time equality. If None, a relative
                 tolerance is used: 1e-12 * max(1, |t|).

        Returns:
            int: The index at which the sample resides after the operation.
        """
        t = float(t)
        v = float(value)
        if not np.isfinite(t) or not np.isfinite(v):
            raise ValueError("add_point: t and value must be finite numbers")
        tol = self._rel_tol(t) if tol is None else float(tol)

        # Bounds check for normalized mode (use scalar -> array path)
        self._check_normalized_bounds(name, np.asarray([v], dtype=np.float64))

        if name not in self._channels:
            self._channels[name] = (
                np.asarray([t], dtype=np.float64),
                np.asarray([v], dtype=np.float64),
            )
            return 0

        times, values = self._channels[name]
        i = int(np.searchsorted(times, t))

        # Check neighbor(s) for duplicate within tolerance
        if i > 0 and abs(times[i - 1] - t) <= tol:
            values[i - 1] = v  # replace
            return i - 1
        if i < times.size and abs(times[i] - t) <= tol:
            values[i] = v  # replace
            return i

        # True insertion
        new_t = np.insert(times, i, t)
        new_v = np.insert(values, i, v)
        self._channels[name] = (
            np.ascontiguousarray(new_t, dtype=np.float64),
            np.ascontiguousarray(new_v, dtype=np.float64),
        )
        return i

    # ------------------------------------------ New Modifier ------------------------------------------

    def add_series(
        self,
        name: str,
        times: Iterable[float],
        values: Iterable[float],
        *,
        tol: float | None = None,
    ) -> int:
        """
        Merge a batch of samples into channel 'name' (sorted, deduplicated).

        Behavior:
            • Incoming (times, values) are coerced to 1D float64 and sorted by time.
            • Duplicate times within the incoming batch are collapsed by last value wins.
            • Merged with existing channel via replacement-on-duplicate (within tol).
            • Keeps strictly increasing times after merge.

        Args:
            name: Channel name.
            times: Iterable of sample times (seconds).
            values: Iterable of sample values (units depend on mode).
            tol: Absolute tolerance for time equality. If None, per-sample relative
                 tolerances are used (1e-12 * max(1, |t|)) during merging.

        Returns:
            int: New total number of samples in the channel after merge.
        """
        t_in = np.asarray(list(times), dtype=np.float64)
        v_in = np.asarray(list(values), dtype=np.float64)

        if t_in.ndim != 1 or v_in.ndim != 1 or t_in.size != v_in.size:
            raise ValueError("add_series: times and values must be 1D of equal length")
        if t_in.size == 0:
            return len(self._channels.get(name, (np.empty(0), np.empty(0)))[0])
        if not np.isfinite(t_in).all() or not np.isfinite(v_in).all():
            raise ValueError("add_series: times/values must be finite")
        # Normalized bounds check for batch
        self._check_normalized_bounds(name, v_in)

        # Sort incoming by time and collapse duplicates (last wins)
        order = np.argsort(t_in)
        t_sorted = t_in[order]
        v_sorted = v_in[order]

        # Collapse duplicates within incoming using a simple pass
        T, V = [t_sorted[0]], [v_sorted[0]]
        for k in range(1, t_sorted.size):
            if abs(t_sorted[k] - T[-1]) <= (tol if tol is not None else self._rel_tol(t_sorted[k])):
                V[-1] = v_sorted[k]  # overwrite last
            else:
                T.append(t_sorted[k]); V.append(v_sorted[k])
        t_new = np.asarray(T, dtype=np.float64)
        v_new = np.asarray(V, dtype=np.float64)

        if name not in self._channels:
            # No existing data: accept batch (times need not be strictly increasing initially—now they are)
            self._channels[name] = (
                np.ascontiguousarray(t_new, dtype=np.float64),
                np.ascontiguousarray(v_new, dtype=np.float64),
            )
            return t_new.size

        # Merge with existing channel
        t_old, v_old = self._channels[name]
        # Fast path: if all new times are after old, append
        if t_new[0] > t_old[-1] + (tol if tol is not None else self._rel_tol(t_new[0])):
            t_merged = np.concatenate([t_old, t_new])
            v_merged = np.concatenate([v_old, v_new])
            self._channels[name] = (
                np.ascontiguousarray(t_merged, dtype=np.float64),
                np.ascontiguousarray(v_merged, dtype=np.float64),
            )
            return t_merged.size

        # Prepend fast path: all new times before old
        if t_new[-1] < t_old[0] - (tol if tol is not None else self._rel_tol(t_new[-1])):
            t_merged = np.concatenate([t_new, t_old])
            v_merged = np.concatenate([v_new, v_old])
            self._channels[name] = (
                np.ascontiguousarray(t_merged, dtype=np.float64),
                np.ascontiguousarray(v_merged, dtype=np.float64),
            )
            return t_merged.size

        # General merge with replacement on duplicate (two-pointer)
        i = j = 0
        out_t: list[float] = []
        out_v: list[float] = []
        while i < t_old.size and j < t_new.size:
            ti, tj = t_old[i], t_new[j]
            rtol_i = tol if tol is not None else self._rel_tol(tj)
            if tj < ti - rtol_i:
                out_t.append(tj); out_v.append(v_new[j]); j += 1
            elif ti < tj - rtol_i:
                out_t.append(ti); out_v.append(v_old[i]); i += 1
            else:
                # duplicate time: take NEW value (replace)
                out_t.append(ti)   # or tj; equal within tol
                out_v.append(v_new[j])
                i += 1; j += 1
        # tail
        if i < t_old.size:
            out_t.extend(t_old[i:].tolist()); out_v.extend(v_old[i:].tolist())
        if j < t_new.size:
            out_t.extend(t_new[j:].tolist()); out_v.extend(v_new[j:].tolist())

        t_merged = np.ascontiguousarray(out_t, dtype=np.float64)
        v_merged = np.ascontiguousarray(out_v, dtype=np.float64)
        # Ensure strict monotonicity (defensive; should already hold)
        if not np.all(np.diff(t_merged) > 0.0):
            raise RuntimeError("Merged times are not strictly increasing (internal error)")
        self._channels[name] = (t_merged, v_merged)
        return t_merged.size

    # ------------------------------------------ New Modifier ------------------------------------------

    def remove_point(self, name: str, t: float, *, tol: float | None = None) -> int:
        """
        Remove the sample closest to time 't' within tolerance.

        Args:
            name: Channel name.
            t: Time of the sample to remove.
            tol: Absolute tolerance; if None uses relative tolerance.

        Returns:
            int: Index removed.

        Raises:
            KeyError: If channel doesn't exist or no sample matches within tolerance.
        """
        if name not in self._channels:
            raise KeyError(f"Unknown channel '{name}'")
        times, values = self._channels[name]
        t = float(t)
        tol = self._rel_tol(t) if tol is None else float(tol)

        idx = int(np.searchsorted(times, t))
        cand = []
        if idx > 0: cand.append(idx - 1)
        if idx < times.size: cand.append(idx)
        for i in cand:
            if abs(times[i] - t) <= tol:
                self._channels[name] = (
                    np.ascontiguousarray(np.delete(times, i), dtype=np.float64),
                    np.ascontiguousarray(np.delete(values, i), dtype=np.float64),
                )
                return i
        raise KeyError(f"remove_point: no sample at t≈{t} within tol={tol}")

    # ------------------------------------------ New Modifier ------------------------------------------

    def remove_at(self, name: str, index: int) -> int:
        """
        Remove the sample at integer 'index' from channel 'name'.

        Returns:
            int: The index that was removed.

        Raises:
            KeyError / IndexError: If channel missing or index OOB.
        """
        if name not in self._channels:
            raise KeyError(f"Unknown channel '{name}'")
        times, values = self._channels[name]
        n = times.size
        if index < 0 or index >= n:
            raise IndexError(f"remove_at: index {index} out of bounds for N={n}")
        self._channels[name] = (
            np.ascontiguousarray(np.delete(times, index), dtype=np.float64),
            np.ascontiguousarray(np.delete(values, index), dtype=np.float64),
        )
        return index
    
    # ------------------------------------------ New Modifier ------------------------------------------

    def clear_channel(self, name: str) -> None:
        """
        Remove an entire channel.
        """
        if name in self._channels:
            del self._channels[name]
    
    # ------------------------------------------ New Modifier ------------------------------------------


    def clear(self) -> None:
        """
        Remove all channels (empty series).
        """
        self._channels.clear()

    # ------------------------------------------ New Modifier ------------------------------------------

    def shift_time(self, dt: float) -> None:
        """
        Shift all channel times by a constant offset `dt` (seconds).
        Useful to re-anchor schedules so the earliest time is 0, etc.
        """
        dt = float(dt)
        for name, (t, v) in self._channels.items():
            self._channels[name] = (np.ascontiguousarray(t + dt, dtype=np.float64), v)

    # ------------------------------------------ New Modifier ------------------------------------------

    def normalize_start(self) -> None:
        """
        Shift all channels so that the global earliest sample occurs at t = 0.
        Does nothing if the series is empty.
        """
        if not self._channels:
            return
        t0 = min(t[0] for (t, _) in self._channels.values())
        self.shift_time(-t0)

    # ------------------------------------------ Queries ------------------------------------------

    def value_at(
        self,
        tq: Iterable[float] | np.ndarray,
        channels: Iterable[str] | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Interpolate each requested channel at query times 'tq'.

        Policy:
            • Linear interpolation within a channel's domain.
            • Outside the domain (t < t_min or t > t_max): returns 0.0.
            • If a requested channel does not exist: returns an all-zeros array.

        Args:
            tq: Query times (1D array-like).
            channels: Iterable of channel names to evaluate. If None, uses all present channels.

        Returns:
            dict[name, np.ndarray]: Each value is a float64 array with shape (len(tq),).
        """
        tq_arr = np.asarray(list(tq), dtype=np.float64)
        if tq_arr.ndim != 1:
            raise ValueError("value_at: tq must be 1D")
        if not np.isfinite(tq_arr).all():
            raise ValueError("value_at: tq contains NaN/Inf")

        names = list(channels) if channels is not None else list(self._channels.keys())
        out: dict[str, np.ndarray] = {}

        # If no channels requested and none present, return empty dict
        if not names and not self._channels:
            return out

        for name in names:
            if name not in self._channels:
                out[name] = np.zeros_like(tq_arr, dtype=np.float64)
                continue
            t, v = self._channels[name]
            # np.interp: linear interp; left/right supply values outside domain
            y = np.interp(tq_arr, t, v, left=0.0, right=0.0).astype(np.float64, copy=False)
            out[name] = y
        return out

    # ------------------------------------------ New Query ------------------------------------------

    def value_at_channel(self, name: str, tq: Iterable[float] | np.ndarray) -> np.ndarray:
        """
        Convenience: interpolate a single channel at times 'tq' with the same policy as 'value_at'.
        Robust to generators: materializes tq once.
        """
        tq_arr = np.asarray(list(tq), dtype=np.float64)
        return self.value_at(tq_arr, channels=[name]).get(name, np.zeros_like(tq_arr, dtype=np.float64))

    # ------------------------------------------ Visualization ------------------------------------------

    def plot(self, *, figsize: tuple[float, float] = (8.0, 6.0)):
        """
        Visualize the time series as three stacked subplots (shared x-axis):

            top:    Omega (Ω)
            middle: Delta (Δ)
            bottom: Phi   (Φ)

        Style / semantics:
            - Serif fonts (Times), STIX math text (matches AtomRegister.plot).
            - Blue dots for input samples; solid orange line connecting them.
            - Orange dashed lines for default values (=0) only where the union
            time domain extends beyond a channel's own domain.
            - If a channel is absent entirely, draw an orange dashed 0-line
            across the union domain (no text).
            - Title and legend are figure-anchored (never overlap subplots).

        Args:
            figsize: Figure size in inches.

        Returns:
            (fig, axes): matplotlib Figure and an array of three Axes (Ω, Δ, Φ).
        """
        import matplotlib.pyplot as plt
        import numpy as np

        # Match plotting style used in atoms.py
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Times New Roman"]
        plt.rcParams["mathtext.fontset"] = "stix"

        # Case-insensitive channel lookup
        key_by_lower = {k.lower(): k for k in self._channels.keys()}
        def get_tv(canon: str):
            k = key_by_lower.get(canon)
            return None if k is None else self._channels[k]  # (times, values)

        # NOTE: use manual layout (no constrained_layout) so we can reserve top space
        fig, axes = plt.subplots(3, 1, sharex=True, figsize=figsize, constrained_layout=False)
        ax_omega, ax_delta, ax_phi = axes

        # Union domain across all present channels (for default-line extents)
        tmin = tmax = None
        if self._channels:
            tmin = min(t[0] for (t, _) in self._channels.values())
            tmax = max(t[-1] for (t, _) in self._channels.values())

        def plot_channel(ax, canon_name: str, ylabel: str, normalized_range_hint: bool):
            tv = get_tv(canon_name.lower())

            if tv is None:
                # Channel missing: show default(0) across union domain if we have one
                if tmin is not None and tmax is not None and np.isfinite([tmin, tmax]).all() and tmax >= tmin:
                    ax.hlines(0.0, tmin, tmax, colors="orange", linestyles="dashed",
                            label="default values (0 outside domain)")
                ax.grid(True, alpha=0.3)
                ax.set_ylabel(ylabel, fontsize=12)
                if self.mode == "normalized" and normalized_range_hint:
                    if canon_name == "omega":
                        ax.set_ylim(-0.05, 1.05)
                    elif canon_name == "delta":
                        ax.set_ylim(-1.05, 1.05)
                return

            # Channel present
            t, v = tv
            ax.plot(t, v, "o", color="blue", label="input values")
            ax.plot(t, v, "-", color="orange", linewidth=1.5, label="input curve")

            # Add default(0) segments only where the union domain extends beyond this channel
            if tmin is not None and tmax is not None and t.size:
                if t[0] > tmin:
                    ax.hlines(0.0, tmin, t[0], colors="orange", linestyles="dashed",
                            label="default values (0 outside domain)")
                if t[-1] < tmax:
                    ax.hlines(0.0, t[-1], tmax, colors="orange", linestyles="dashed",
                            label="default values (0 outside domain)")

            ax.grid(True, alpha=0.3)
            ax.set_ylabel(ylabel, fontsize=12)
            if self.mode == "normalized" and normalized_range_hint and t.size:
                if canon_name == "omega":
                    ax.set_ylim(-0.05, 1.05)
                elif canon_name == "delta":
                    ax.set_ylim(-1.05, 1.05)

        # Axis labels & title text
        if self.mode == "absolute":
            y_omega = r"$\Omega$"
            y_delta = r"$\Delta$"
            y_phi   = r"$\Phi$ (rad)"
            title_str = "TimeSeries (Absolute Values)"
        else:
            y_omega = r"$\Omega_{\mathrm{env}}$"
            y_delta = r"$\Delta_{\mathrm{env}}$"
            y_phi   = r"$\Phi_{\mathrm{env}}$"
            title_str = "TimeSeries (Normalized Values)"

        plot_channel(ax_omega, "omega", y_omega, normalized_range_hint=(self.mode == "normalized"))
        plot_channel(ax_delta, "delta", y_delta, normalized_range_hint=(self.mode == "normalized"))
        plot_channel(ax_phi,   "phi",   y_phi,   normalized_range_hint=False)

        # X label + limits
        ax_phi.set_xlabel(r"time $t$ (s)", fontsize=12)
        if tmin is not None and tmax is not None and np.isfinite([tmin, tmax]).all():
            span = max(tmax - tmin, 1e-15)
            ax_phi.set_xlim(tmin - 0.02 * span, tmax + 0.02 * span)

        # ----- Figure-anchored title & legend (stable placement, no overlaps) -----
        # Title near the top of the figure
        fig.suptitle(title_str, fontsize=14, y=0.98)

        # Build a single legend from the axis with the richest set of labels; de-duplicate
        handles_labels = [
            ax_omega.get_legend_handles_labels(),
            ax_delta.get_legend_handles_labels(),
            ax_phi.get_legend_handles_labels(),
        ]
        handles, labels = max(handles_labels, key=lambda hl: len(hl[1]))
        seen = set(); dedup_h, dedup_l = [], []
        for h, l in zip(handles, labels):
            if l not in seen:
                seen.add(l); dedup_h.append(h); dedup_l.append(l)

        if dedup_h:
            # Legend sits below the title, above the axes
            fig.legend(dedup_h, dedup_l,
                    loc="upper center", bbox_to_anchor=(0.5, 0.93),
                    ncol=3, frameon=True, fontsize=11)

        # Reserve vertical headroom so neither title nor legend overlaps the plots
        fig.subplots_adjust(top=0.84)

        return fig, axes

    