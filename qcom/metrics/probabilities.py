# qcom/metrics/probabilities.py
"""
qcom.metrics.probabilities
==========================

Probability-distribution utilities and derived summaries.

Purpose
-------
Provide helpers for working with classical probability dictionaries coming
from measurement outcomes (e.g., {bitstring: p}), including:
- Scalar queries like the cumulative probability up to a threshold.
- Construction of cumulative distribution “steps” on linear or log-spaced grids.
- The N(p) diagnostic (mass within a multiplicative neighborhood around p).
- Convenience routine to obtain |ψ|^2 of an eigenstate in the computational basis.

Notes
-----
- Inputs are assumed to be dictionaries mapping bit-strings (e.g., "0101")
  to probabilities that (usually) sum to 1.0.
- `get_eigenstate_probabilities` depends on the static eigen-solver.

Endianness
----------
- Throughout, the default convention is MSB ↔ site 0 (big-endian), matching the
  solver’s basis ordering. Several routines offer `msb_site0=False` to flip to
  “LSB ↔ site 0” (little-endian): this is implemented by reversing formatted
  bitstrings only (no reindexing of the state vector).

Log Bases
---------
- Unless otherwise stated, log grids and δ-widths are in base-10 (“log10”).
  Where appropriate, a `base` (for grids) or `log_base` (for N(p)) argument is
  included for forward compatibility; defaults preserve previous behavior.
"""

from __future__ import annotations

import numpy as np
from .._internal.progress import ProgressManager
from ..solvers.static import find_eigenstate, as_linear_operator


# -------------------- Cumulative probability at a threshold --------------------

def cumulative_probability_at_value(binary_dict: dict[str, float], value: float) -> float:
    """
    Sum probabilities in `binary_dict` that are ≤ `value`.

    Args:
        binary_dict: Mapping of state bit-strings to probabilities.
        value: Threshold probability.

    Returns:
        float: Sum of all p such that p ≤ value.
    """
    probabilities = np.array(list(binary_dict.values()), dtype=float)
    return float(np.sum(probabilities[probabilities <= value]))


# -------------------- Cumulative distribution (linear or log grid) --------------------

# -------------------- Cumulative distribution (unique or user-supplied grid) --------------------

def cumulative_distribution(
    binary_dict: dict[str, float],
    grid: np.ndarray | list[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a cumulative-distribution curve from {state -> probability}.

    Modes
    -----
    - grid=None:
        Return the “unique-probability” step function:
        x = sorted unique probabilities plus a final 1.0,
        y = cumulative mass at those x, normalized so y[-1] = 1.
    - grid=array-like:
        Treat `grid` as the exact probability thresholds at which to evaluate
        the CDF. For each x in the (sorted) grid, y[i] = Σ_{p ≤ x} p. The
        resulting y is normalized by the final running sum so that y[-1] = 1.0.
        (This matches prior behavior where the fixed grid’s last point set the
        normalization.)

    Args:
        binary_dict: Keys are bit-strings; values must sum to 1.0 (within fp tol).
        grid: Array-like of probability thresholds in (0, 1]; may be linear or
              log-spaced. If you want a log grid, pass something like:
                  # log10 grid
                  np.logspace(np.log10(p_min), np.log10(1.0), num=N, base=10)
                  # natural-log grid
                  np.logspace(np.log(p_min), np.log(1.0), num=N, base=np.e)
                  # base-2 grid
                  np.logspace(np.log2(p_min), 0.0, num=N, base=2)
              (Here p_min is the smallest nonzero probability you wish to include.)

    Returns:
        (x_axis, y_axis): 1D arrays of equal length.

    Notes
    -----
    - Normalization uses a robust guard in case of tiny numerical drift and
      guarantees y[-1] == 1.0 even if round-off leaves the final sum at zero.
    - If `grid` contains unsorted values, they are internally sorted ascending.
    """
    if not binary_dict:
        raise ValueError("cumulative_distribution: input dict is empty")

    probs = np.array(list(binary_dict.values()), dtype=float)
    if not np.isclose(np.sum(probs), 1.0):
        raise ValueError("cumulative_distribution: input values must sum to 1.0")

    if grid is None:
        sorted_p = np.sort(probs)
        unique_p, counts = np.unique(sorted_p, return_counts=True)
        cumulative = np.cumsum(unique_p * counts, dtype=float)
        denom = cumulative[-1] if cumulative[-1] > 0 else np.finfo(float).tiny
        cumulative /= denom  # ensure final = 1.0 (robust to numerical quirks)
        x_axis = np.append(unique_p, 1.0)
        y_axis = np.append(cumulative, 1.0)
        return x_axis, y_axis

    # Use user-supplied probability grid
    x_axis = np.asarray(grid, dtype=float).reshape(-1)
    if x_axis.size < 1:
        raise ValueError("cumulative_distribution: provided grid is empty")
    if np.any(x_axis <= 0) or np.any(x_axis > 1.0):
        raise ValueError("cumulative_distribution: grid values must lie in (0, 1]")

    # Ensure ascending thresholds
    x_axis = np.sort(x_axis)

    sorted_p = np.sort(probs)
    n = sorted_p.size
    idx = 0
    running_sum = 0.0
    y_axis = np.zeros_like(x_axis, dtype=float)

    for i, c in enumerate(x_axis):
        while idx < n and sorted_p[idx] <= c:
            running_sum += sorted_p[idx]
            idx += 1
        y_axis[i] = running_sum

    # Robust normalization: guarantee y[-1] == 1.0
    denom = running_sum if running_sum > 0.0 else np.finfo(float).tiny
    y_axis = y_axis / denom
    y_axis[-1] = 1.0

    return x_axis, y_axis


# -------------------- N(p) diagnostics (bulk) --------------------

def compute_N_of_p_all(
    probabilities: dict[str, float] | np.ndarray | list[float],
    p_delta: float = 0.1,
    show_progress: bool = False,
    *,
    log_base: float = 10.0,
) -> tuple[np.ndarray, list[float]]:
    """
    Efficiently compute N(p) for each unique nonzero probability.

    Definition (heuristic)
    ----------------------
    For a given p, define a log window
        [ p * (log_base)^(-δ/2),  p * (log_base)^(+δ/2) ]
    and compute
        N(p) = (Σ_{q in window} q) / ( (upper - lower) * p ).

    Args:
        probabilities: Dict of probabilities or an array-like of probabilities.
        p_delta: Width δ in log space (default log_base = 10 → log10 units).
        show_progress: Whether to show a simple progress indicator.
        log_base: Base of the logarithm used to define the multiplicative window.
                  Defaults to 10.0 to match previous (log10) behavior.

    Returns:
        (unique_probs, N_values)
    """
    if isinstance(probabilities, dict):
        probabilities = list(probabilities.values())

    if log_base <= 0.0 or log_base == 1.0:
        raise ValueError("compute_N_of_p_all: 'log_base' must be > 0 and != 1.0")

    probs = np.array(probabilities, dtype=float)
    probs = probs[probs > 0.0]
    if probs.size == 0:
        return np.array([], dtype=float), []

    sorted_probs = np.sort(probs)
    cumulative_probs = np.cumsum(sorted_probs)
    unique_probs = np.unique(sorted_probs)

    def compute_single_N(p: float) -> float:
        # window edges in the chosen base
        b = float(log_base)
        scale = b ** (p_delta / 2.0)
        lower = p / scale
        upper = p * scale

        lower_idx = int(np.searchsorted(sorted_probs, lower, side="left"))
        upper_idx = int(np.searchsorted(sorted_probs, upper, side="right"))
        sigma_lower = cumulative_probs[lower_idx - 1] if lower_idx > 0 else 0.0
        sigma_upper = cumulative_probs[upper_idx - 1] if upper_idx > 0 else 0.0

        width = (upper - lower) * p
        return float((sigma_upper - sigma_lower) / (width if width > 0 else np.finfo(float).tiny))

    N_values: list[float] = []
    total = len(unique_probs)

    with (
        ProgressManager.progress("Computing N(p)", total_steps=total)
        if show_progress
        else ProgressManager.dummy_context()
    ):
        for i, p in enumerate(unique_probs):
            N_values.append(compute_single_N(float(p)))
            if show_progress:
                ProgressManager.update_progress(i + 1)

    return unique_probs, N_values


# -------------------- N(p) diagnostic (single p) --------------------

def compute_N_of_p(
    p: float,
    sorted_probs: np.ndarray,
    cumulative_probs: np.ndarray,
    p_delta: float = 0.1,
    *,
    log_base: float = 10.0,
) -> float:
    """
    Compute N(p) using precomputed sorted/cumulative arrays.

    Args:
        p: Probability at which to evaluate N(p).
        sorted_probs: Sorted array of nonzero probabilities.
        cumulative_probs: Cumulative sum of `sorted_probs`.
        p_delta: Width δ in log space (default log_base = 10 → log10 units).
        log_base: Base of the logarithm used to define the multiplicative window.
                  Defaults to 10.0 to match previous (log10) behavior.

    Returns:
        float: N(p) value.
    """
    if p <= 0.0:
        return 0.0
    if log_base <= 0.0 or log_base == 1.0:
        raise ValueError("compute_N_of_p: 'log_base' must be > 0 and != 1.0")

    b = float(log_base)
    scale = b ** (p_delta / 2.0)
    lower = p / scale
    upper = p * scale

    lower_idx = int(np.searchsorted(sorted_probs, lower, side="left"))
    upper_idx = int(np.searchsorted(sorted_probs, upper, side="right"))

    sigma_lower = cumulative_probs[lower_idx - 1] if lower_idx > 0 else 0.0
    sigma_upper = cumulative_probs[upper_idx - 1] if upper_idx > 0 else 0.0

    width = (upper - lower) * p
    denom = width if width > 0 else np.finfo(float).tiny
    return float((sigma_upper - sigma_lower) / denom)


# -------------------- Eigenstate probabilities (computational basis) --------------------

def get_eigenstate_probabilities(
    hamiltonian,
    *,
    state_index: int = 0,
    show_progress: bool = False,
    drop_tol: float = 0.0,
    msb_site0: bool = True,
) -> dict[str, float]:
    """
    Compute |ψ|^2 in the computational basis for a chosen eigenstate.

    Works with:
      - NumPy dense arrays
      - SciPy sparse matrices
      - SciPy LinearOperator
      - qcom.hamiltonians.BaseHamiltonian instances

    Args:
        hamiltonian: Hamiltonian-like object
        state_index: Which eigenstate to use (0 = ground).
        show_progress: Whether to display progress updates.
        drop_tol: if > 0, omit entries with prob <= drop_tol.
        msb_site0: If True (default), format bitstrings with MSB ↔ site 0
                   (big-endian). If False, flip to “LSB ↔ site 0” by reversing
                   formatted bitstrings (little-endian).

    Returns:
        dict[str, float]: Mapping "bitstring" -> probability.
    """
    # --- Determine Hilbert dimension robustly (no densification) ---
    dim = None
    try:
        # Fast path for BaseHamiltonian
        from ..hamiltonians.base import BaseHamiltonian
        if isinstance(hamiltonian, BaseHamiltonian):
            dim = hamiltonian.hilbert_dim
    except Exception:
        pass

    if dim is None:
        # Dense, sparse, LinearOperator → normalize via as_linear_operator
        _, _, dim = as_linear_operator(hamiltonian)

    # Check it’s a power of two and get qubit count
    n_qubits = int(np.round(np.log2(dim)))
    if (1 << n_qubits) != dim:
        raise ValueError(f"Hilbert dim {dim} is not a power of 2; cannot map to bitstrings.")

    hilbert_dim = dim
    total_steps = 4 + hilbert_dim
    step = 0

    with (
        ProgressManager.progress("Computing Eigenstate Probabilities", total_steps)
        if show_progress
        else ProgressManager.dummy_context()
    ):
        # --- 1) Eigenstate (do NOT densify; find_eigenstate handles all backends) ---
        _, chosen_state = find_eigenstate(
            hamiltonian,
            state_index=state_index,
            show_progress=show_progress,
        )
        step += 1
        if show_progress:
            ProgressManager.update_progress(min(step, total_steps))

        # --- 2) Probabilities |ψ|^2 and normalize ---
        psi = np.asarray(chosen_state).reshape(-1)
        probabilities = np.abs(psi) ** 2
        s = float(probabilities.sum())
        if s > 0.0:
            probabilities /= s
        step += 1
        if show_progress:
            ProgressManager.update_progress(min(step, total_steps))

        # --- 3) Build mapping with selectable endianness ---
        fmt = "{:0" + str(n_qubits) + "b}"
        state_prob_dict: dict[str, float] = {}
        thr = float(drop_tol)
        for i in range(hilbert_dim):
            p = float(probabilities[i])
            if p > thr:
                bitstring = fmt.format(i)
                if not msb_site0:
                    bitstring = bitstring[::-1]
                state_prob_dict[bitstring] = p
            step += 1
            if show_progress and (step % 1024 == 0 or i == hilbert_dim - 1):
                # update periodically to avoid excessive overhead
                ProgressManager.update_progress(min(step, total_steps))

        # --- 4) Final update to hit total_steps exactly ---
        if show_progress and step < total_steps:
            ProgressManager.update_progress(total_steps)

    return state_prob_dict


# -------------------- Get State Probabilities --------------------

def statevector_to_probabilities(
    psi: np.ndarray,
    *,
    msb_site0: bool = True,
    drop_tol: float = 0.0,
) -> dict[str, float]:
    """
    Convert a state vector into a dictionary of computational-basis probabilities.

    Args
    ----
    psi : np.ndarray
        State vector of shape (2^N,). Entries are complex amplitudes.
    msb_site0 : bool, optional
        If True (default), format bitstrings with MSB ↔ site 0 (big-endian).
        If False, flip to “LSB ↔ site 0” by reversing formatted bitstrings
        (little-endian). This matches the option used elsewhere.
    drop_tol : float, optional
        If > 0, omit entries with probability ≤ drop_tol.

    Returns
    -------
    dict[str, float]
        Dictionary mapping bitstrings (e.g. "00", "01", "10", "11") to probabilities.
        Probabilities are floats in [0,1] and sum to 1 (up to numerical error).

    Notes
    -----
    - Convention by default: MSB ↔ site 0 (the same ordering as used by the solver).
      Example: for N=2, index 2 (binary "10") means site 0 excited, site 1 ground.
    - Set `msb_site0=False` to use little-endian labeling instead.
    """
    psi = np.asarray(psi, dtype=np.complex128).reshape(-1)
    dim = psi.shape[0]
    if dim == 0 or (dim & (dim - 1)) != 0:
        raise ValueError(f"State vector length {dim} is not a power of 2.")

    N = int(np.log2(dim))
    probs = np.abs(psi) ** 2
    s = float(probs.sum())
    if s > 0.0:
        probs /= s  # renormalize just in case

    out: dict[str, float] = {}
    thr = float(drop_tol)
    for i, p in enumerate(probs):
        if p > thr:
            bitstring = format(i, f"0{N}b")
            if not msb_site0:
                bitstring = bitstring[::-1]
            out[bitstring] = float(p)
    return out