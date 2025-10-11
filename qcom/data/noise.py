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

# ------------------------------------------ Imports ------------------------------------------

import random


# ------------------------------------------ Public API ------------------------------------------

def introduce_error(
    data: dict[str, int],
    ground_rate: float = 0.01,
    excited_rate: float = 0.08,
) -> dict[str, int]:
    """
    Apply an independent readout-error channel to classical bitstring counts.

    For each shot (i.e., for each counted occurrence of a bitstring), every bit
    is flipped independently according to:
        - P(0 → 1) = ground_rate
        - P(1 → 0) = excited_rate

    This is a *Monte Carlo* (sampling) implementation: integer counts are
    expanded implicitly shot-by-shot and re-accumulated after flips.

    Args:
        data:
            Mapping from bitstring -> integer count (e.g., {"0101": 12, ...}).
        ground_rate:
            Probability of flipping a measured '0' to '1' (0 ≤ rate ≤ 1).
        excited_rate:
            Probability of flipping a measured '1' to '0' (0 ≤ rate ≤ 1).

    Returns:
        dict[str, int]: New dictionary of bitstring counts after simulated readout errors.

    Notes:
        • If you want a *deterministic* transformation on a probability vector
          (without Monte Carlo), consider implementing a confusion-matrix-based
          linear map in a future probability-domain API.
        • To make outcomes reproducible in tests, set a seed in the caller:
              random.seed(1234)
    """
    new_counts: dict[str, int] = {}

    for state, count in data.items():
        # Process each "shot" (occurrence) of this bitstring
        for _ in range(count):
            bits = list(state)
            # Flip each bit independently according to its channel
            for i, b in enumerate(bits):
                if b == "1":
                    if random.random() < excited_rate:
                        bits[i] = "0"
                else:  # b == "0"
                    if random.random() < ground_rate:
                        bits[i] = "1"
            new_state = "".join(bits)
            new_counts[new_state] = new_counts.get(new_state, 0) + 1

    return new_counts