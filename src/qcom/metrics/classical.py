"""
qcom.metrics.classical
======================

Classical information-theoretic measures on probability distributions.

Purpose
-------
Provide basic Shannon-type metrics from classical information theory,
computed directly from measurement probability dictionaries.

Functions
---------
- compute_shannon_entropy(prob_dict, total_prob=1):
    Shannon entropy of a probability distribution.

- compute_reduced_shannon_entropy(prob_dict, configuration, target_region):
    Reduced Shannon entropy of a subsystem defined by a site configuration.

- compute_mutual_information(prob_dict, configuration, total_count=1):
    Classical mutual information between two regions.

- compute_conditional_entropy(prob_dict, configuration, total_count=1):
    Conditional entropy of one region given the other.
"""

from __future__ import annotations

from typing import Iterable, Sequence
import numpy as np
from qcom.core import MutualInformationResult
from .bitstrings import marginalize_bitstring_distribution, sort_bitstring_distribution


def _validate_probabilities(bitstring_probabilities: dict[str, float]) -> None:
    if not isinstance(bitstring_probabilities, dict):
        raise TypeError("Expected 'bitstring_probabilities' to be a dict[str, float].")
    if len(bitstring_probabilities) == 0:
        raise ValueError("Empty probability dictionary.")
    for bitstring, probability in bitstring_probabilities.items():
        if (
            not isinstance(bitstring, str)
            or any(char not in "01" for char in bitstring)
            or len(bitstring) == 0
        ):
            raise ValueError(f"Invalid bitstring key {bitstring!r}.")
        if probability < 0:
            raise ValueError(f"Negative probability for key {bitstring!r}: {probability}.")


def _entropy_from_values(values: Iterable[float], *, total: float | None, base: float) -> float:
    """
    Core entropy with optional normalization and arbitrary log base.
    If total is None, uses sum(values). Zeros are skipped safely.
    """
    vals = np.asarray(list(values), dtype=float)
    if vals.size == 0:
        return 0.0
    tot = float(vals.sum()) if total is None else float(total)
    if tot <= 0.0:
        return 0.0

    p = vals / tot
    # filter strictly positive to avoid log(0)
    p = p[p > 0.0]
    if p.size == 0:
        return 0.0

    if base == np.e:
        return float(-np.sum(p * np.log(p)))
    else:
        return float(-np.sum(p * (np.log(p) / np.log(base))))


def _indices_from_configuration(configuration: Sequence[int], target_region: int) -> list[int]:
    if configuration is None:
        raise ValueError("configuration must be provided when 'indices' is not given.")
    if target_region not in (0, 1):
        raise ValueError("target_region must be 0 or 1.")
    return [i for i, reg in enumerate(configuration) if int(reg) == target_region]


def _marginal(prob_dict: dict[str, float], indices: Sequence[int]) -> dict[str, float]:
    reduced = marginalize_bitstring_distribution(prob_dict, indices)
    return sort_bitstring_distribution(reduced)


def compute_shannon_entropy(
    prob_dict: dict[str, float],
    total_prob: float | None = 1.0,
    *,
    base: float = np.e,
) -> float:
    """
    Computes the Shannon entropy H(P) = -Σ p log_b p.

    Args:
        prob_dict: Mapping bitstring → probability (not necessarily normalized).
        total_prob: If None, infer from sum(prob_dict.values()); otherwise use this
                    as the normalization constant.
        base: Logarithm base (e for nats [default], 2 for bits).

    Returns:
        float: Shannon entropy.
    """
    _validate_probabilities(prob_dict)
    return _entropy_from_values(prob_dict.values(), total=total_prob, base=base)


def compute_reduced_shannon_entropy(
    prob_dict: dict[str, float],
    configuration: Sequence[int] | None = None,
    target_region: int = 0,
    *,
    indices: Sequence[int] | None = None,
    base: float = np.e,
) -> float:
    """
    Computes the reduced Shannon entropy for a given subsystem.

    You can specify the subsystem either by:
      • (configuration, target_region)
      • indices (explicit list of bit positions, 0 = MSB)

    Args:
        prob_dict: Mapping bitstring → probability (not necessarily normalized).
        configuration: Binary list specifying a bipartition (0 for A, 1 for B).
        target_region: Region to compute entropy for (0 = A, 1 = B).
        indices: Alternative explicit indices of the kept subsystem (MSB=0).
        base: Logarithm base.

    Returns:
        float: Reduced Shannon entropy H(P_subsystem).
    """
    _validate_probabilities(prob_dict)

    if indices is None:
        idxs = _indices_from_configuration(configuration, target_region)  # type: ignore[arg-type]
    else:
        idxs = list(int(i) for i in indices)
        if len(idxs) == 0:
            return 0.0

    reduced = _marginal(prob_dict, idxs)
    total = sum(reduced.values())
    return _entropy_from_values(reduced.values(), total=total, base=base)


def compute_mutual_information(
    prob_dict: dict[str, float],
    configuration: Sequence[int] | None = None,
    *,
    a_indices: Sequence[int] | None = None,
    b_indices: Sequence[int] | None = None,
    base: float = np.e,
    return_components: bool = False,
) -> float | MutualInformationResult:
    """
    Computes classical mutual information I(A:B) = H(A) + H(B) - H(AB).

    You can specify A/B either by:
      • configuration (0 for A, 1 for B), or
      • a_indices, b_indices (explicit lists, MSB=0).

    Args:
        prob_dict: Mapping bitstring → probability (not necessarily normalized).
        configuration: Binary list specifying the bipartition.
        a_indices: Explicit indices for region A.
        b_indices: Explicit indices for region B.
        base: Logarithm base.

    Returns:
        float:
            Mutual information I(A:B) by default.
        MutualInformationResult:
            If ``return_components=True``, returns I(A:B) plus H(A), H(B),
            and H(AB).
    """
    _validate_probabilities(prob_dict)

    # Decide indices
    if a_indices is None or b_indices is None:
        if configuration is None:
            raise ValueError("Provide either (configuration) or both (a_indices, b_indices).")
        a_indices = [i for i, v in enumerate(configuration) if int(v) == 0]
        b_indices = [i for i, v in enumerate(configuration) if int(v) == 1]
    else:
        a_indices = list(int(i) for i in a_indices)
        b_indices = list(int(i) for i in b_indices)

    # Entropies of marginals and joint
    h_ab = compute_shannon_entropy(prob_dict, total_prob=None, base=base)

    a_marginal = _marginal(prob_dict, a_indices)
    h_a = _entropy_from_values(a_marginal.values(), total=sum(a_marginal.values()), base=base)

    b_marginal = _marginal(prob_dict, b_indices)
    h_b = _entropy_from_values(b_marginal.values(), total=sum(b_marginal.values()), base=base)

    mutual_information = h_a + h_b - h_ab
    if return_components:
        return MutualInformationResult(
            mutual_information=mutual_information,
            h_a=h_a,
            h_b=h_b,
            h_ab=h_ab,
        )
    return mutual_information


def compute_conditional_entropy(
    prob_dict: dict[str, float],
    configuration: Sequence[int] | None = None,
    *,
    a_indices: Sequence[int] | None = None,
    b_indices: Sequence[int] | None = None,
    base: float = np.e,
) -> float:
    """
    Computes the conditional entropy H(A|B) = H(AB) - H(B).

    You can specify A/B either by:
      • configuration (0 for A, 1 for B), or
      • a_indices, b_indices (explicit lists, MSB=0). (Only B is strictly required,
        but both are accepted for symmetry with `compute_mutual_information`.)

    Args:
        prob_dict: Mapping bitstring → probability (not necessarily normalized).
        configuration: Binary list specifying the bipartition.
        a_indices: Explicit indices for region A (unused in formula but allowed).
        b_indices: Explicit indices for region B.
        base: Logarithm base.

    Returns:
        float: H(A|B)
    """
    _validate_probabilities(prob_dict)

    # Determine B indices
    if b_indices is None:
        if configuration is None:
            raise ValueError("Provide either (configuration) or b_indices for region B.")
        b_indices = [i for i, v in enumerate(configuration) if int(v) == 1]
    else:
        b_indices = list(int(i) for i in b_indices)

    h_ab = compute_shannon_entropy(prob_dict, total_prob=None, base=base)

    b_marginal = _marginal(prob_dict, b_indices)
    h_b = _entropy_from_values(b_marginal.values(), total=sum(b_marginal.values()), base=base)

    return h_ab - h_b
