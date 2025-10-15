"""
Noise models for classical post-processing of measurement data.

Goal
----
Provide simple, composable noise channels that operate on classical
bitstring counts/probabilities produced by experiments or simulations.
This module starts with a readout-error model and is designed to be
extended with additional noise processes.

Scope (current)
---------------
• Readout error (independent bit flips per site):
  - ground_rate:  P(read 0 → 1)
  - excited_rate: P(read 1 → 0)
  Applies shot-by-shot Monte Carlo to integer-count datasets.

Design notes
-----------
• Pure classical post-processing: inputs are dict[str, int] of counts.
• Stateless, functional API – easy to compose (call one noise function
  after another) and to test deterministically when seeding 'random'.
• No silent renormalization: counts remain counts; probabilities remain
  probabilities (up to the caller).

Future directions
-----------------
• Additional channels:
  - symmetric/asymmetric bit-flip channels on probability vectors;
  - erasure/dropout models; crosstalk models (pairwise correlated flips);
  - SPAM models with calibration-derived confusion matrices.
• A subpackage structure (e.g., qcom/data/noise/...) grouping multiple
  noise models with shared utilities.
"""

from __future__ import annotations

# ------------------------------------------ Imports ------------------------------------------
from typing import Mapping, Sequence
import random

# ------------------------------------------ Helpers ------------------------------------------

def _infer_num_sites(data: Mapping[str, int]) -> int:
    if not data:
        raise ValueError("Empty dataset: cannot infer number of sites.")
    lengths = {len(k) for k in data.keys()}
    if len(lengths) != 1:
        raise ValueError(f"All bitstrings must have equal length; got lengths {sorted(lengths)}.")
    return lengths.pop()

def _broadcast_rates(
    rate: float | Sequence[float] | Mapping[int, float],
    N: int,
    *,
    default: float = 0.0,
    name: str = "rate",
) -> list[float]:
    # scalar → broadcast
    if isinstance(rate, (int, float)):
        r = float(rate)
        if not (0.0 <= r <= 1.0):
            raise ValueError(f"{name} must be in [0,1]; got {r}.")
        return [r] * N
    # sequence → per-site
    if isinstance(rate, Sequence) and not isinstance(rate, (str, bytes)):
        if len(rate) != N:
            raise ValueError(f"{name} sequence length {len(rate)} != N={N}.")
        out = []
        for i, r in enumerate(rate):
            rr = float(r)
            if not (0.0 <= rr <= 1.0):
                raise ValueError(f"{name}[{i}] not in [0,1]: {rr}.")
            out.append(rr)
        return out
    # mapping → sparse overrides
    if isinstance(rate, Mapping):
        out = [default] * N
        for i, r in rate.items():
            if not (0 <= int(i) < N):
                raise ValueError(f"{name} dict key {i} out of range [0,{N-1}].")
            rr = float(r)
            if not (0.0 <= rr <= 1.0):
                raise ValueError(f"{name}[{i}] not in [0,1]: {rr}.")
            out[int(i)] = rr
        return out
    raise TypeError(f"{name}: expected float, sequence, or mapping; got {type(rate).__name__}.")

def _confusion_matrices_from_rates(
    ground_rate: float | Sequence[float] | Mapping[int, float],
    excited_rate: float | Sequence[float] | Mapping[int, float],
    N: int,
) -> list:
    import numpy as np
    p01 = _broadcast_rates(ground_rate,  N, name="ground_rate")
    p10 = _broadcast_rates(excited_rate, N, name="excited_rate")
    mats = []
    for i in range(N):
        mats.append(
            np.array([[1.0 - p01[i], p10[i]],
                      [p01[i],       1.0 - p10[i]]], dtype=np.float32)
        )
    return mats

# ------------------------------------------ Public API ------------------------------------------

def introduce_error(
    data: dict[str, int],
    ground_rate: float | Sequence[float] | Mapping[int, float] = 0.01,  # P(0→1)
    excited_rate: float | Sequence[float] | Mapping[int, float] = 0.08, # P(1→0)
    *,
    seed: int | None = None,
) -> dict[str, int]:
    """
    Monte-Carlo readout error on integer counts with **global or per-site** rates.

    Args:
        data: dict[bitstring, count], all bitstrings must share the same length.
        ground_rate: P(measured '0' flips to '1'); float, per-site sequence, or {site: value} mapping.
        excited_rate: P(measured '1' flips to '0'); float, per-site sequence, or {site: value} mapping.
        seed: optional RNG seed for reproducibility.

    Returns:
        dict[str, int]: New counts after simulated readout errors.
    """
    N = _infer_num_sites(data)
    p01 = _broadcast_rates(ground_rate,  N, name="ground_rate")
    p10 = _broadcast_rates(excited_rate, N, name="excited_rate")

    rng = random.Random(seed)
    out: dict[str, int] = {}

    for bitstr, cnt in data.items():
        bits = list(bitstr)
        for _ in range(int(cnt)):
            flip = bits[:]  # copy
            for i, b in enumerate(bits):
                if b == "0":
                    if rng.random() < p01[i]:
                        flip[i] = "1"
                else:  # "1"
                    if rng.random() < p10[i]:
                        flip[i] = "0"
            new = "".join(flip)
            out[new] = out.get(new, 0) + 1
    return out

def m3_mitigate_counts_from_rates(
    counts: dict[str, int],
    *,
    ground_rate: float | Sequence[float] | Mapping[int, float] = 0.01,
    excited_rate: float | Sequence[float] | Mapping[int, float] = 0.08,
    qubits: list[int] | None = None,
) -> dict[str, float]:
    """
    Build per-qubit confusion matrices from {p01, p10} and run mthree mitigation.

    Returns:
        dict[str, float]: mitigated quasi-probabilities (clipped to ≥0 and re-normalized).
    """
    try:
        import mthree
        import numpy as np
    except Exception as e:
        raise ImportError(
            "mthree is required for mitigation: pip install qiskit-addon-mthree"
        ) from e

    if not counts:
        return {}

    # Infer N and default qubit order [0..N-1]
    lengths = {len(k) for k in counts}
    if len(lengths) != 1:
        raise ValueError(f"All bitstrings must have equal length; got lengths {sorted(lengths)}.")
    N = lengths.pop()
    meas_qubits = qubits if qubits is not None else list(range(N))

    matrices = _confusion_matrices_from_rates(ground_rate, excited_rate, N)

    mit = mthree.M3Mitigation()
    mit.cals_from_matrices(matrices)

    int_counts = {k: int(v) for k, v in counts.items()}
    mitigated = mit.apply_correction(int_counts, meas_qubits)

    out = {k: float(v) for k, v in mitigated.items()}
    s = sum(out.values())
    if s > 0.0:
        # clip tiny negatives from quasi-probs and renormalize
        out = {k: max(0.0, v) for k, v in out.items()}
        s2 = sum(out.values())
        if s2 > 0.0:
            out = {k: v / s2 for k, v in out.items()}
    return out