"""
QCOM solver entry points.
"""

from qcom.core import EvolutionResult, SpectrumResult

from .dynamic import evolve_state
from .static import (
    as_linear_operator,
    eigensolve,
    find_eigenstate,
    full_dense_spectrum,
    ground_state,
)

__all__ = [
    "EvolutionResult",
    "SpectrumResult",
    "as_linear_operator",
    "eigensolve",
    "evolve_state",
    "find_eigenstate",
    "full_dense_spectrum",
    "ground_state",
]
