"""
QCOM
====

Quantum Computation (QCOM) is a Python package originally developed as part of Avi Kaufman’s 2025 honors thesis.
It provides tools for analyzing quantum systems, including Hamiltonian construction, time evolution solvers,
and various metrics for evaluating quantum states and operations.

Author
------
Avi Kaufman

Version
-------
0.2.1
"""

__version__ = "0.2.1"
__author__ = "Avi Kaufman"
__all__ = [
    "controls",
    "data",
    "hamiltonians",
    "io",
    "metrics",
    "solvers",
    "lattice_register",
]

# Re-export subpackages for convenience
from . import controls
from . import data
from . import hamiltonians
from . import io
from . import metrics
from . import solvers
from . import lattice_register