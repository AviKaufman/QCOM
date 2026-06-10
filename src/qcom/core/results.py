"""
Typed result and measurement containers for QCOM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

import numpy as np


def _validate_bitstring_mapping(values: Mapping[str, int | float], *, name: str) -> int:
    if not isinstance(values, Mapping):
        raise TypeError(f"{name} must be a mapping from bitstring to value.")
    if not values:
        raise ValueError(f"{name} must not be empty.")

    lengths: set[int] = set()
    for bitstring, value in values.items():
        if not isinstance(bitstring, str) or not bitstring or any(c not in "01" for c in bitstring):
            raise ValueError(f"Invalid bitstring key {bitstring!r}.")
        if value < 0:
            raise ValueError(f"Negative value for key {bitstring!r}: {value}.")
        lengths.add(len(bitstring))
    if len(lengths) != 1:
        raise ValueError(f"{name} keys must all have the same bitstring length.")
    return lengths.pop()


@dataclass(frozen=True)
class MutualInformationResult:
    """Components of classical mutual information."""

    mutual_information: float
    h_a: float
    h_b: float
    h_ab: float


@dataclass(frozen=True)
class CountsData:
    """Typed raw-count measurement dataset."""

    counts: Mapping[str, int]
    shots: int | None = None
    n_sites: int | None = None
    source: str | None = None

    def __post_init__(self) -> None:
        clean = {str(k): int(v) for k, v in self.counts.items()}
        inferred_sites = _validate_bitstring_mapping(clean, name="counts")
        inferred_shots = sum(clean.values())

        shots = inferred_shots if self.shots is None else int(self.shots)
        if shots < 0:
            raise ValueError("shots must be non-negative.")
        if shots != inferred_shots:
            raise ValueError(f"shots={shots} does not match sum(counts)={inferred_shots}.")

        n_sites = inferred_sites if self.n_sites is None else int(self.n_sites)
        if n_sites != inferred_sites:
            raise ValueError(f"n_sites={n_sites} does not match bitstring length={inferred_sites}.")

        object.__setattr__(self, "counts", MappingProxyType(clean))
        object.__setattr__(self, "shots", shots)
        object.__setattr__(self, "n_sites", n_sites)

    def to_dict(self) -> dict[str, int]:
        """Return a mutable copy of the raw counts."""
        return dict(self.counts)


@dataclass(frozen=True)
class ProbabilityData:
    """Typed bitstring-probability dataset."""

    probabilities: Mapping[str, float]
    n_sites: int | None = None
    normalized: bool = True
    source: str | None = None

    def __post_init__(self) -> None:
        clean = {str(k): float(v) for k, v in self.probabilities.items()}
        inferred_sites = _validate_bitstring_mapping(clean, name="probabilities")

        n_sites = inferred_sites if self.n_sites is None else int(self.n_sites)
        if n_sites != inferred_sites:
            raise ValueError(f"n_sites={n_sites} does not match bitstring length={inferred_sites}.")

        total = sum(clean.values())
        if self.normalized and not np.isclose(total, 1.0, rtol=0.0, atol=1e-9):
            raise ValueError(f"normalized probabilities must sum to 1.0; got {total}.")

        object.__setattr__(self, "probabilities", MappingProxyType(clean))
        object.__setattr__(self, "n_sites", n_sites)

    def to_dict(self) -> dict[str, float]:
        """Return a mutable copy of the probabilities."""
        return dict(self.probabilities)


@dataclass(frozen=True)
class EvolutionResult:
    """Structured result for time evolution."""

    final_state: np.ndarray
    times: np.ndarray | None = None
    states: tuple[np.ndarray, ...] | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "final_state", np.asarray(self.final_state, dtype=np.complex128))
        if self.times is not None:
            object.__setattr__(self, "times", np.asarray(self.times, dtype=np.float64))
        if self.states is not None:
            object.__setattr__(
                self,
                "states",
                tuple(np.asarray(state, dtype=np.complex128) for state in self.states),
            )
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True)
class SpectrumResult:
    """Structured result for eigenvalue calculations."""

    eigenvalues: np.ndarray
    eigenvectors: np.ndarray | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "eigenvalues", np.asarray(self.eigenvalues))
        if self.eigenvectors is not None:
            object.__setattr__(self, "eigenvectors", np.asarray(self.eigenvectors))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
