# qcom/solvers/dynamic.py
"""
Dynamic solver — generic time evolution under time-dependent Hamiltonians.

This module evolves a quantum state under H(t) using full matrix exponentials.
It is model-agnostic: you provide
  • a TimeSeries of control channels,
  • an adapter that maps sampled channel values → H(t),
  • an initial state vector.

We intentionally do NOT expose Trotter modes here. If you need more scalable
methods, use a dedicated algorithmic solver (e.g., TEBD) elsewhere in the package.

Typical usage
-------------
>>> from qcom.controls.time_series import TimeSeries
>>> from qcom.controls.adapters.rydberg import RydbergAdapter
>>> from qcom.solvers.dynamic import evolve_state
>>> ts = TimeSeries(...); adapter = RydbergAdapter(...)
>>> psi_T, out = evolve_state(ts, adapter, psi0, n_steps=400, record=True)

Notes
-----
• Accepts dense ndarray or CSR/CSC sparse matrices from the adapter.
• Uses SciPy for expm (dense or sparse).
• Progress display integrates with `ProgressManager` if available.
"""

# ------------------------------------------ Imports ------------------------------------------

from __future__ import annotations

import numpy as np
from typing import Iterable, Mapping, Protocol, runtime_checkable

# Optional SciPy backends (dense and sparse expm)
try:
    import scipy.linalg as _sla  # dense expm
except Exception:  # pragma: no cover
    _sla = None

try:
    import scipy.sparse as _sp
    import scipy.sparse.linalg as _spl
except Exception:  # pragma: no cover
    _sp = None
    _spl = None

# Progress manager is optional; if not present we no-op
try:
    from qcom.progress import ProgressManager
except Exception:  # pragma: no cover
    class _DummyPM:
        @staticmethod
        def progress(*args, **kwargs):
            from contextlib import nullcontext
            return nullcontext()
        @staticmethod
        def dummy_context():
            from contextlib import nullcontext
            return nullcontext()
        @staticmethod
        def update_progress(*args, **kwargs):
            pass
    ProgressManager = _DummyPM()  # type: ignore


# ------------------------------------------ Adapter Protocol ------------------------------------------

@runtime_checkable
class ControlAdapter(Protocol):
    """
    Minimal protocol an adapter must follow.

    Required
    --------
    • required_channels : tuple[str, ...]
        Names of channels to sample from the TimeSeries (case-sensitive).
    • hamiltonian_at(t: float, controls: Mapping[str, float]) -> ArrayLike
        Return the full Hermitian matrix H(t) (dense ndarray or sparse CSR/CSC).

    Optional
    --------
    • dimension : int
        If provided and nonzero, used for early validation of the initial state size.
    """

    @property
    def required_channels(self) -> tuple[str, ...]: ...
    def hamiltonian_at(self, t: float, controls: Mapping[str, float]): ...
    @property
    def dimension(self) -> int: ...


# ------------------------------------------ Utilities ------------------------------------------

def _is_sparse(A) -> bool:
    return _sp is not None and _sp.issparse(A)

def _expm(A):
    """
    Matrix exponential expm(A) supporting dense and sparse inputs.
    Returns the same container type (dense ndarray or sparse CSC).
    """
    is_sp = _is_sparse(A)
    if is_sp:
        if _spl is None:
            raise RuntimeError("SciPy sparse is required for sparse expm but is not available.")
        # Ensure CSC for efficient solves inside expm
        A = A.tocsc(copy=False)
        return _spl.expm(A)  # returns sparse CSC
    else:
        if _sla is None:
            raise RuntimeError("SciPy is required for dense expm but is not available.")
        return _sla.expm(np.asarray(A, dtype=np.complex128))

def _apply_unitary(U, psi):
    """
    Apply U @ psi where U may be dense or sparse. psi is dense vector.
    Returns a dense complex128 vector.
    """
    if _is_sparse(U):
        out = U.dot(psi)
    else:
        out = U @ psi
    return np.asarray(out, dtype=np.complex128)

def _normalize(psi):
    nrm = np.linalg.norm(psi)
    if nrm == 0.0:
        return psi
    return psi / nrm


# ------------------------------------------ Public API ------------------------------------------

def evolve_state(
    time_series,
    adapter: ControlAdapter,
    psi0: np.ndarray,
    *,
    n_steps: int | None = None,
    times: Iterable[float] | None = None,
    normalize_each_step: bool = True,
    record: bool = False,
    show_progress: bool = True,
):
    """
    Evolve a state under a time-dependent Hamiltonian using full expm of H(t_mid)*dt.

    Time grid
    ---------
    • Provide exactly one of:
        - `times` : explicit monotonically increasing array-like including endpoints, OR
        - `n_steps`: number of uniform steps across the union domain of the TimeSeries.
    • The simulation window is inferred from `time_series.domain()`; there is no t_span argument.

    Args
    ----
    time_series:
        A `TimeSeries` providing control channels. It may contain more channels than required;
        only `adapter.required_channels` are sampled.
    adapter:
        Object implementing the ControlAdapter protocol (see above).
    psi0:
        Initial state vector (1D, complex128). Length must match Hilbert dimension.
    n_steps:
        Number of uniform steps across the TimeSeries union domain. Use when `times` is not given.
    times:
        Explicit monotonically increasing array-like of times including both endpoints.
        If provided, `n_steps` must be None.
    normalize_each_step:
        If True, renormalize the state vector after each step (numerical hygiene).
    record:
        If True, return a trajectory dict with "times" and "states".
    show_progress:
        If True, display a progress bar using ProgressManager.

    Returns
    -------
    psi_T : np.ndarray
        Final state at the last time in the grid.
    out : dict
        Metadata; if `record` is True, contains:
          - "times": np.ndarray of shape (M,)
          - "states": list[np.ndarray] (length M)
    """
    # ----- Validate initial state ------------------------------------------------
    psi = np.asarray(psi0, dtype=np.complex128).reshape(-1)
    dim = psi.shape[0]
    if hasattr(adapter, "dimension"):
        try:
            adim = int(adapter.dimension)  # type: ignore[attr-defined]
            if adim and adim != dim:
                raise ValueError(
                    f"Initial state dimension {dim} does not match adapter.dimension={adim}."
                )
        except Exception:
            pass

    # ----- Build time grid -------------------------------------------------------
    if (times is None) == (n_steps is None):
        raise ValueError("Provide exactly one of 'n_steps' or 'times'.")

    if times is not None:
        t_grid = np.asarray(list(times), dtype=np.float64)
        if t_grid.ndim != 1 or t_grid.size < 2:
            raise ValueError("'times' must be a 1D array with at least two entries (t0, t1).")
    else:
        # derive from the TimeSeries union domain
        t0, t1 = time_series.domain()
        if n_steps < 1:
            raise ValueError("'n_steps' must be >= 1.")
        t_grid = np.linspace(float(t0), float(t1), int(n_steps) + 1, dtype=np.float64)

    # ----- Determine which channels to sample -----------------------------------
    req = tuple(getattr(adapter, "required_channels", ()))
    if not req:
        # Fall back to sampling everything present
        req = tuple(time_series.channel_names)  # type: ignore[attr-defined]

    # Prepare recording
    traj_times = None
    traj_states = None
    if record:
        traj_times = t_grid.copy()
        traj_states = [psi.copy()]

    # Progress: one update per time step
    total_steps = t_grid.size - 1
    with (ProgressManager.progress("Time evolution", total_steps) if show_progress
          else ProgressManager.dummy_context()):
        # ----- Main loop ---------------------------------------------------------
        for s in range(total_steps):
            t_a, t_b = float(t_grid[s]), float(t_grid[s + 1])
            dt = t_b - t_a
            # Sample controls at the subinterval midpoint (piecewise-constant per subinterval)
            t_mid = 0.5 * (t_a + t_b)
            controls = {
                name: float(val[0])
                for name, val in time_series.value_at([t_mid], channels=req).items()
            }  # dict[str, np.ndarray] -> scalar

            # Full H(t_mid) → U = exp(-i H dt)
            H = adapter.hamiltonian_at(t_mid, controls)  # type: ignore[attr-defined]
            # Materialize as sparse CSR if adapter didn't already
            if _is_sparse(H):
                A = H
            else:
                A = np.asarray(H, dtype=np.complex128)
            U = _expm((-1j * dt) * (_sp.csr_matrix(A) if _is_sparse(A) else A))
            psi = _apply_unitary(U, psi)

            if normalize_each_step:
                psi = _normalize(psi)

            if record:
                traj_states.append(psi.copy())

            if show_progress:
                ProgressManager.update_progress(min(s + 1, total_steps))

    out = {}
    if record:
        out["times"] = traj_times
        out["states"] = traj_states
    return psi, out