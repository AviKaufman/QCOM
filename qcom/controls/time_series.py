"""
TimeSeries — generic container for time-dependent control envelopes

Goal
----
Provide a transparent, NumPy-centric container for time-dependent control
signals used to evolve QCOM Hamiltonians (or any simulator). The class enforces
a clear, minimal contract: a set of named channels, each defined by an explicit
(t, value) schedule. The object is responsible only for interpolation and
storage; it does not embed model-specific or site-dependent scaling.

Key guarantees
--------------
• Channels are independent: each channel has its own strictly increasing time
  axis and values. The global domain is the union of all channel domains.
• Storage is float64 NumPy arrays. Insertion keeps arrays strictly sorted.
• Interpolation is linear between sample points.
• Queries outside a channel’s domain (before first or after last sample)
  return 0.0 for that channel.
• Unspecified channels default to a constant 0.0 envelope.
• Duplicate times within a channel are disallowed; inserting at an existing
  time (within tolerance) replaces the previous value (no duplicates).

Design philosophy
-----------------
• Pure container: TimeSeries does not know about a particular Hamiltonian
  (e.g., Rydberg) or any site-level parameters. It only provides envelopes.
  Site dependence and model mapping are applied later (e.g., inside an
  evolution routine that consumes this object).
• Modes:
  - "absolute": values are interpreted as physical quantities by the caller.
  - "normalized": values are dimensionless; any physical scaling is applied
    by the caller. (No built-in bounds are enforced by TimeSeries.)
• Optional constraints: users may register per-channel validators to enforce
  application-specific bounds (e.g., clamp to [0,1]) without hard-coding
  physics into this container.

Editing & mutability
--------------------
• Build schedules incrementally:
  - add_point(name, t, value): insert/replace a single sample at the correct place.
  - add_series(name, times, values): merge multiple samples (sorted, de-duplicated).
• Deletion and clearing are supported:
  - remove_point, remove_at, clear_channel, clear.
• All operations preserve strict time monotonicity.

Primary use cases
-----------------
• Author arbitrary time series manually or from logs.
• Incrementally build/edit schedules with add/remove operations.
• Interpolate channel values at arbitrary query times during simulation.
• Provide envelopes for Hamiltonian evolution; mapping to operators happens
  outside this class.

Future extensions (non-breaking)
--------------------------------
• Alternate interpolation schemes (e.g., spline).
• Convenience constructors (ramps, windows, sinusoids).
• Import/export (JSON, CSV, experiment logs).
• Optional plotting helpers living outside this generic container.
"""

import numpy as np
from typing import Literal, Mapping
from types import MappingProxyType
from collections.abc import Iterable

# ------------------------------------------ TimeSeries Class ------------------------------------------

class TimeSeries:
    """
    Generic container for time-dependent control envelopes.

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
                  - "absolute": values are in physical units (caller interprets).
                  - "normalized": values are dimensionless; caller applies scaling.
            **channels:
                Optional initial schedules as keyword args: name=(times, values),
                where both are 1D, same length, finite, strictly increasing times.
                Times may be any real numbers (negative allowed); only strict
                monotonicity is required.
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

            # Store (no physics-specific bounds)
            self._channels[name] = (
                np.ascontiguousarray(t, dtype=np.float64),
                np.ascontiguousarray(v, dtype=np.float64),
            )


    # ------------------------------------------ Properties ------------------------------------------
    
    @property
    def channels(self) -> Mapping[str, tuple[np.ndarray, np.ndarray]]:
        """
        Read-only snapshot of all channels.

        Returns:
            Mapping[str, (times, values)]:
                A mapping where both arrays are read-only NumPy views.
                Times are strictly increasing, values are float64.
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
        """
        Tuple of channel names currently present.

        Returns:
            tuple[str, ...]: Channel names in insertion order.
        """
        return tuple(self._channels.keys())

    
    # ------------------------------------------ New Property ------------------------------------------

    @property
    def is_empty(self) -> bool:
        """
        Whether the TimeSeries has no channels.

        Returns:
            bool: True if empty, False otherwise.
        """
        return not self._channels


    # ------------------------------ Read-only accessors ------------------------------

    def __len__(self) -> int:
        """
        Number of channels currently stored.

        Returns:
            int: Count of channels.
        """
        return len(self._channels)

    # ------------------------------ New Read-only ------------------------------

    def has_channel(self, name: str) -> bool:
        """
        Check whether a channel exists.

        Args:
            name: Channel name (case-sensitive).

        Returns:
            bool: True if the channel is present, False otherwise.
        """
        return name in self._channels

    # ------------------------------ New Read-only ------------------------------

    def times(self, name: str) -> np.ndarray:
        """
        Read-only view of the time grid for a channel.

        Args:
            name: Channel name.

        Returns:
            np.ndarray: 1D float64 array of strictly increasing times.

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
            np.ndarray: 1D float64 array aligned with `times(name)`.

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
        Return the time domain of a channel, or of the entire TimeSeries.

        Args:
            name: If provided, return domain for that channel only.
                If None, return the union domain across all channels.

        Returns:
            (t_min, t_max): Floats delimiting the domain.

        Raises:
            KeyError: If `name` is provided and the channel does not exist.
            ValueError: If no channels are present (union domain undefined).
        """
        if name is not None:
            try:
                t = self._channels[name][0]
            except KeyError:
                raise KeyError(f"Unknown channel '{name}'") from None
            return float(t[0]), float(t[-1])

        if not self._channels:
            raise ValueError("domain(): no channels present (empty TimeSeries).")

        tmins = [ch[0][0] for ch in self._channels.values()]
        tmaxs = [ch[0][-1] for ch in self._channels.values()]
        return float(min(tmins)), float(max(tmaxs))

    # ------------------------------ Representation ------------------------------

    def __repr__(self) -> str:
        """
        Concise summary string for debugging/inspection.

        Returns:
            str: A one-line summary including mode and channel info.
        """
        parts = []
        for name, (t, _) in self._channels.items():
            parts.append(f"{name}[L={t.size}, {t[0]:.3e}→{t[-1]:.3e}s]")
        body = ", ".join(parts) if parts else "∅"
        mode = self.mode
        return f"TimeSeries(mode={mode!r}, channels={body})"

    # ------------------------------ Utilities ------------------------------

    @staticmethod
    def _rel_tol(t: float) -> float:
        """
        Relative time tolerance used to detect 'the same sample time'.

        Args:
            t: Reference time.

        Returns:
            float: Tolerance = 1e-12 * max(1, |t|).
        """
        return 1e-12 * max(1.0, abs(float(t)))

    # ------------------------------ New Utility ------------------------------

    def _check_normalized_bounds(self, name: str, values: np.ndarray) -> None:
        """
        Enforce normalized-mode bounds for well-known channels.
        No-op in absolute mode.

        Args:
            name: Channel name (case-insensitive check for 'omega'/'delta').
            values: Array of sample values to validate.

        Raises:
            ValueError: If bounds are violated for the given channel.
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
        # Phi intentionally unbounded

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
            # return existing length if channel exists; else 0
            return len(self._channels.get(name, (np.empty(0), np.empty(0)))[0])
        if not np.isfinite(t_in).all() or not np.isfinite(v_in).all():
            raise ValueError("add_series: times/values must be finite")

        # Normalized bounds check for batch
        self._check_normalized_bounds(name, v_in)

        # Sort incoming by time and collapse duplicates (last wins)
        order = np.argsort(t_in)
        t_sorted = t_in[order]
        v_sorted = v_in[order]

        T, V = [t_sorted[0]], [v_sorted[0]]
        for k in range(1, t_sorted.size):
            same = abs(t_sorted[k] - T[-1]) <= (tol if tol is not None else self._rel_tol(t_sorted[k]))
            if same:
                V[-1] = v_sorted[k]  # overwrite last
            else:
                T.append(t_sorted[k]); V.append(v_sorted[k])
        t_new = np.asarray(T, dtype=np.float64)
        v_new = np.asarray(V, dtype=np.float64)

        if name not in self._channels:
            # No existing data: accept batch
            self._channels[name] = (
                np.ascontiguousarray(t_new, dtype=np.float64),
                np.ascontiguousarray(v_new, dtype=np.float64),
            )
            return t_new.size

        # Merge with existing channel
        t_old, v_old = self._channels[name]

        # Fast append: all new times after old
        if t_new[0] > t_old[-1] + (tol if tol is not None else self._rel_tol(t_new[0])):
            t_merged = np.concatenate([t_old, t_new])
            v_merged = np.concatenate([v_old, v_new])
            self._channels[name] = (
                np.ascontiguousarray(t_merged, dtype=np.float64),
                np.ascontiguousarray(v_merged, dtype=np.float64),
            )
            return t_merged.size

        # Fast prepend: all new times before old
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
            rtol = tol if tol is not None else self._rel_tol(tj)
            if tj < ti - rtol:
                out_t.append(tj); out_v.append(v_new[j]); j += 1
            elif ti < tj - rtol:
                out_t.append(ti); out_v.append(v_old[i]); i += 1
            else:
                # duplicate time: take NEW value (replace)
                out_t.append(ti)   # or tj; equal within tol
                out_v.append(v_new[j])
                i += 1; j += 1

        # tails
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
            • If a requested channel does not exist (or is empty): returns an all-zeros array.

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
            tv = self._channels.get(name, None)
            if tv is None:
                out[name] = np.zeros_like(tq_arr, dtype=np.float64)
                continue

            t, v = tv
            # Handle an (unexpected) empty channel defensively with zeros
            if t.size == 0:
                out[name] = np.zeros_like(tq_arr, dtype=np.float64)
                continue

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

    def plot(
        self,
        *,
        style: str = "auto",
        order: Iterable[str] | None = None,
        labels: Mapping[str, str] | None = None,
        y_hints: Mapping[str, tuple[float, float]] | None = None,
        figsize: tuple[float, float] | None = None,
    ):
        """
        Visualize the time series.

        Styles
        ------
        - style="auto"        : If channels are exactly {Omega, Delta, Phi} → use 'rydberg' preset,
                                otherwise fall back to one subplot per present channel.
        - style="per_channel" : One subplot per present channel (generic).
        - style="<preset>"    : Force a preset layout (currently: "rydberg").

        Args:
            style: "auto", "per_channel", or a preset name (e.g., "rydberg").
            order: Optional explicit channel order (also filters to these names).
            labels: Optional channel→y-label overrides.
            y_hints: Optional channel→(ymin, ymax) overrides.
            figsize: Optional (width, height) in inches. If None, chosen automatically.

        Returns:
            (fig, axes): matplotlib Figure and a list of Axes (one per drawn channel).
        """
        import matplotlib.pyplot as plt
        import numpy as np

        # ----- Matplotlib style (match your previous aesthetic) -------------------
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Times New Roman"]
        plt.rcParams["mathtext.fontset"] = "stix"

        # ----- Preset registry (lightweight, extendable) -------------------------
        # Keys are canonical (lowercase) channel names.
        PRESETS = {
            "rydberg": {
                "order": ("omega", "delta", "phi"),
                "labels_abs": {
                    "omega": r"$\Omega$",
                    "delta": r"$\Delta$",
                    "phi":   r"$\Phi\,(\mathrm{rad})$",
                },
                "labels_norm": {
                    "omega": r"$\Omega_{\mathrm{env}}$",
                    "delta": r"$\Delta_{\mathrm{env}}$",
                    "phi":   r"$\Phi_{\mathrm{env}}$",
                },
                # Normalized-mode range hints (optional)
                "norm_y_hints": {
                    "omega": (-0.05, 1.05),
                    "delta": (-1.05, 1.05),
                    # 'phi' intentionally unbounded
                },
                "title_abs": "TimeSeries (Rydberg, Absolute Values)",
                "title_norm": "TimeSeries (Rydberg, Normalized Values)",
            },
        }

        # ----- Helper: canonicalize / choose channels ----------------------------
        # Map canonical lowercase → original key in storage (preserve insertion order)
        lower_to_orig = {k.lower(): k for k in self._channels.keys()}
        present_canon = list(lower_to_orig.keys())

        if not present_canon:
            # No channels to draw
            fig, ax = plt.subplots(figsize=(figsize if figsize else (6.0, 2.0)))
            ax.axis("off")
            ax.set_title("TimeSeries (empty)")
            return fig, [ax]

        # Decide plotting mode / preset
        chosen_preset = None
        if style == "auto":
            # Detect Rydberg by exact set equality
            if set(present_canon) == {"omega", "delta", "phi"}:
                chosen_preset = "rydberg"
            else:
                chosen_preset = None  # per-channel fallback
        elif style == "per_channel":
            chosen_preset = None
        else:
            # Treat any other string as a preset name (if known)
            if style.lower() in PRESETS:
                chosen_preset = style.lower()
            else:
                raise ValueError(f"Unknown plot style/preset: {style!r}")

        # Determine the list of canonical channel names to draw
        if order is not None:
            # Respect user-specified order and filter; match case-insensitively
            desired = [str(n).lower() for n in order]
            canon_list = [c for c in desired if c in lower_to_orig]
        elif chosen_preset is not None:
            preset_order = PRESETS[chosen_preset]["order"]
            canon_list = [c for c in preset_order if c in lower_to_orig]  # skip missing silently
        else:
            # Per-channel generic: use insertion order
            canon_list = list(present_canon)

        if not canon_list:
            # Nothing matched (e.g., order filtered everything out)
            fig, ax = plt.subplots(figsize=(figsize if figsize else (6.0, 2.0)))
            ax.axis("off")
            ax.set_title("TimeSeries (no matching channels)")
            return fig, [ax]

        # ----- Labels & y-limits (merge preset defaults + user overrides) --------
        labels = dict(labels or {})
        y_hints = dict(y_hints or {})

        if chosen_preset is not None:
            preset = PRESETS[chosen_preset]
            # Pick default labels based on mode
            default_labels = preset["labels_norm"] if self.mode == "normalized" else preset["labels_abs"]
            for c in canon_list:
                if c not in labels and c in default_labels:
                    labels[c] = default_labels[c]
            # Normalized y-hints (only fill if not given by user)
            if self.mode == "normalized":
                for c, rng in preset.get("norm_y_hints", {}).items():
                    if c in canon_list and c not in y_hints:
                        y_hints[c] = rng

        # Fallback label if still missing
        for c in canon_list:
            if c not in labels:
                labels[c] = c  # use the channel name itself

        # ----- Global time domain (for dashed default segments & x-limits) -------
        tmin = min(self._channels[lower_to_orig[c]][0][0] for c in present_canon)
        tmax = max(self._channels[lower_to_orig[c]][0][-1] for c in present_canon)
        have_union = np.isfinite([tmin, tmax]).all() and (tmax >= tmin)

        # ----- Figure & axes allocation ------------------------------------------
        n_axes = len(canon_list)
        if figsize is None:
            # Height heuristic: ~2.2in per axis, min 2.0in, max 12in total by default
            h = min(max(2.2 * n_axes, 2.0), 12.0)
            figsize = (8.0, h)

        fig, axes = plt.subplots(n_axes, 1, sharex=True, figsize=figsize, constrained_layout=False)
        if n_axes == 1:
            axes = [axes]  # normalize to list

        # ----- Draw channels ------------------------------------------------------
        for ax, c in zip(axes, canon_list):
            name_orig = lower_to_orig[c]
            t, v = self._channels[name_orig]

            # Main curve and sample markers
            ax.plot(t, v, "o", label="input values")
            ax.plot(t, v, "-", linewidth=1.5, label="input curve")

            # Dashed default(0) segments where union domain extends beyond this channel's domain
            if have_union and t.size:
                if t[0] > tmin:
                    ax.hlines(0.0, tmin, t[0], linestyles="dashed", label="default outside domain")
                if t[-1] < tmax:
                    ax.hlines(0.0, t[-1], tmax, linestyles="dashed", label="default outside domain")

            ax.grid(True, alpha=0.3)
            ax.set_ylabel(labels.get(c, c), fontsize=12)

            # Apply y-range hints if any
            if c in y_hints:
                ymin, ymax = y_hints[c]
                ax.set_ylim(float(ymin), float(ymax))

        # X label + limits
        axes[-1].set_xlabel(r"time $t$ (s)", fontsize=12)
        if have_union:
            span = max(tmax - tmin, 1e-15)
            axes[-1].set_xlim(tmin - 0.02 * span, tmax + 0.02 * span)

        # ----- Title & legend placement ------------------------------------------
        if chosen_preset is not None:
            preset = PRESETS[chosen_preset]
            title_str = preset["title_norm"] if self.mode == "normalized" else preset["title_abs"]
        else:
            title_str = "TimeSeries (Per-Channel)" if style == "per_channel" else "TimeSeries"

        # Put title at very top; we’ll compute its height and then place legend just below it.
        title_obj = fig.suptitle(title_str, fontsize=14, y=1.0, va="bottom")

        # Build a single legend from the axis with the richest labels; de-duplicate
        handles_labels = [ax.get_legend_handles_labels() for ax in axes]
        handles, texts = max(handles_labels, key=lambda hl: len(hl[1]))
        legend_obj = None
        if texts:
            seen = set(); H = []; T = []
            for h, ttxt in zip(handles, texts):
                if ttxt not in seen:
                    seen.add(ttxt); H.append(h); T.append(ttxt)
            # Place legend in figure coords; exact y will be set after we know its height
            legend_obj = fig.legend(
                H, T,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.0),
                bbox_transform=fig.transFigure,
                ncol=min(3, len(T)),
                frameon=True,
                fontsize=11,
            )

        # --- Dynamic top padding so legend always clears the top subplot by the same gap ---
        # We need a renderer to measure title/legend heights.
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

        # Convert bounding boxes to figure-relative units
        def _h_in_figcoords(artist):
            bbox = artist.get_window_extent(renderer).transformed(fig.transFigure.inverted())
            return bbox.height

        title_h = _h_in_figcoords(title_obj) if title_obj is not None else 0.0
        legend_h = _h_in_figcoords(legend_obj) if legend_obj is not None else 0.0

        # Gaps in *figure fraction* for consistent spacing across subplot counts
        gap_title_legend = 0.010  # space between title and legend
        gap_legend_axes  = 0.020  # space between legend and top subplot

        # Compute top fraction
        top = 1.0 - (title_h + (legend_h if legend_obj else 0.0)
                     + (gap_title_legend if legend_obj else 0.0)
                     + gap_legend_axes)

        # Keep within sane bounds in case of huge fonts
        top = max(0.55, min(0.92, top))

        fig.subplots_adjust(top=top)

        return fig, axes