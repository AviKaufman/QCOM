"""
Noise models for classical post-processing of measurement data.

The public helpers operate on bitstring counts produced by experiments or
simulations. The current model is independent readout error: each measured bit
can flip according to per-site or global rates.
"""

from __future__ import annotations

import random
from typing import Mapping, Sequence

from qcom._internal.deprecations import warn_deprecated_alias

__all__ = ["apply_readout_error", "introduce_error", "m3_mitigate_counts_from_rates"]


def _infer_num_sites(counts: Mapping[str, int]) -> int:
    if not counts:
        raise ValueError("Empty dataset: cannot infer number of sites.")
    bitstring_lengths = {len(bitstring) for bitstring in counts.keys()}
    if len(bitstring_lengths) != 1:
        raise ValueError(
            f"All bitstrings must have equal length; got lengths {sorted(bitstring_lengths)}."
        )
    return bitstring_lengths.pop()


def _broadcast_rates(
    rate: float | Sequence[float] | Mapping[int, float],
    num_sites: int,
    *,
    default: float = 0.0,
    name: str = "rate",
) -> list[float]:
    if isinstance(rate, (int, float)):
        rate_value = float(rate)
        if not (0.0 <= rate_value <= 1.0):
            raise ValueError(f"{name} must be in [0,1]; got {rate_value}.")
        return [rate_value] * num_sites

    if isinstance(rate, Sequence) and not isinstance(rate, (str, bytes)):
        if len(rate) != num_sites:
            raise ValueError(f"{name} sequence length {len(rate)} != num_sites={num_sites}.")
        rate_values = []
        for site_index, rate_value in enumerate(rate):
            value = float(rate_value)
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{name}[{site_index}] not in [0,1]: {value}.")
            rate_values.append(value)
        return rate_values

    if isinstance(rate, Mapping):
        rate_values = [default] * num_sites
        for site_index, rate_value in rate.items():
            if not (0 <= int(site_index) < num_sites):
                raise ValueError(f"{name} dict key {site_index} out of range [0,{num_sites - 1}].")
            value = float(rate_value)
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{name}[{site_index}] not in [0,1]: {value}.")
            rate_values[int(site_index)] = value
        return rate_values

    raise TypeError(f"{name}: expected float, sequence, or mapping; got {type(rate).__name__}.")


def _confusion_matrices_from_rates(
    ground_rate: float | Sequence[float] | Mapping[int, float],
    excited_rate: float | Sequence[float] | Mapping[int, float],
    num_sites: int,
) -> list:
    import numpy as np

    p01 = _broadcast_rates(ground_rate, num_sites, name="ground_rate")
    p10 = _broadcast_rates(excited_rate, num_sites, name="excited_rate")
    matrices = []
    for site_index in range(num_sites):
        matrices.append(
            np.array(
                [
                    [1.0 - p01[site_index], p10[site_index]],
                    [p01[site_index], 1.0 - p10[site_index]],
                ],
                dtype=np.float32,
            )
        )
    return matrices


def apply_readout_error(
    counts: dict[str, int],
    ground_rate: float | Sequence[float] | Mapping[int, float] = 0.01,  # P(0->1)
    excited_rate: float | Sequence[float] | Mapping[int, float] = 0.08,  # P(1->0)
    *,
    seed: int | None = None,
) -> dict[str, int]:
    """
    Apply Monte Carlo readout error to integer bitstring counts.

    Args:
        counts: Bitstring counts. All bitstrings must share the same length.
        ground_rate: Probability that a measured "0" flips to "1"; may be a
            scalar, per-site sequence, or `{site: value}` mapping.
        excited_rate: Probability that a measured "1" flips to "0"; may be a
            scalar, per-site sequence, or `{site: value}` mapping.
        seed: Optional RNG seed for reproducibility.

    Returns:
        New counts after simulated readout errors.
    """
    num_sites = _infer_num_sites(counts)
    p01 = _broadcast_rates(ground_rate, num_sites, name="ground_rate")
    p10 = _broadcast_rates(excited_rate, num_sites, name="excited_rate")

    rng = random.Random(seed)
    noisy_counts: dict[str, int] = {}

    for bitstring, count in counts.items():
        bits = list(bitstring)
        for _ in range(int(count)):
            flipped_bits = bits[:]
            for site_index, bit in enumerate(bits):
                if bit == "0":
                    if rng.random() < p01[site_index]:
                        flipped_bits[site_index] = "1"
                elif rng.random() < p10[site_index]:
                    flipped_bits[site_index] = "0"
            noisy_bitstring = "".join(flipped_bits)
            noisy_counts[noisy_bitstring] = noisy_counts.get(noisy_bitstring, 0) + 1

    return noisy_counts


def introduce_error(
    data: dict[str, int],
    ground_rate: float | Sequence[float] | Mapping[int, float] = 0.01,
    excited_rate: float | Sequence[float] | Mapping[int, float] = 0.08,
    *,
    seed: int | None = None,
) -> dict[str, int]:
    """Deprecated compatibility alias for `apply_readout_error`."""
    warn_deprecated_alias("introduce_error", "apply_readout_error")
    return apply_readout_error(
        data,
        ground_rate=ground_rate,
        excited_rate=excited_rate,
        seed=seed,
    )


def m3_mitigate_counts_from_rates(
    counts: dict[str, int],
    *,
    ground_rate: float | Sequence[float] | Mapping[int, float] = 0.01,
    excited_rate: float | Sequence[float] | Mapping[int, float] = 0.08,
    qubits: list[int] | None = None,
) -> dict[str, float]:
    """
    Build per-qubit confusion matrices from rates and run mthree mitigation.

    Returns:
        Mitigated quasi-probabilities, clipped to nonnegative values and
        renormalized when possible.
    """
    try:
        import mthree
    except Exception as exc:
        raise ImportError(
            "mthree is required for mitigation: pip install qiskit-addon-mthree"
        ) from exc

    if not counts:
        return {}

    bitstring_lengths = {len(bitstring) for bitstring in counts}
    if len(bitstring_lengths) != 1:
        raise ValueError(
            f"All bitstrings must have equal length; got lengths {sorted(bitstring_lengths)}."
        )
    num_sites = bitstring_lengths.pop()
    measurement_qubits = qubits if qubits is not None else list(range(num_sites))

    matrices = _confusion_matrices_from_rates(ground_rate, excited_rate, num_sites)

    mitigation = mthree.M3Mitigation()
    mitigation.cals_from_matrices(matrices)

    integer_counts = {bitstring: int(value) for bitstring, value in counts.items()}
    mitigated = mitigation.apply_correction(integer_counts, measurement_qubits)

    mitigated_probabilities = {bitstring: float(value) for bitstring, value in mitigated.items()}
    total_probability = sum(mitigated_probabilities.values())
    if total_probability > 0.0:
        mitigated_probabilities = {
            bitstring: max(0.0, value) for bitstring, value in mitigated_probabilities.items()
        }
        clipped_total = sum(mitigated_probabilities.values())
        if clipped_total > 0.0:
            mitigated_probabilities = {
                bitstring: value / clipped_total
                for bitstring, value in mitigated_probabilities.items()
            }

    return mitigated_probabilities
