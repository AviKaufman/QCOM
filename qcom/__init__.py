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

# import specific functions 

# controls
from .controls.time_series import TimeSeries
# controls.adapters.rydberg
from .controls.adapters.rydberg import RydbergAdapter

# data
# data.noise
from .data.noise import introduce_error
from .data.noise import m3_mitigate_counts_from_rates
# data.ops
from .data.ops import normalize_to_probabilities
from .data.ops import truncate_probabilities
from .data.ops import print_most_probable_data
# data.sampling
from .data.sampling import sample_data
from .data.sampling import combine_datasets

# hamiltonians
# hamiltonians.rydberg
from .hamiltonians.rydberg import build_rydberg

# io
# io.aquila
from .io.aquila import parse_json
# io.parquet
from .io.parquet import parse_parquet
# io.text
from .io.text import parse_file
from .io.text import save_data

# metrics
# metrics.classical
from .metrics.classical import compute_shannon_entropy
from .metrics.classical import compute_mutual_information
from .metrics.classical import compute_conditional_entropy
from .metrics.classical import compute_reduced_shannon_entropy
# metrics.entanglement
from .metrics.entanglement import von_neumann_entropy_from_state
# metrics.probabilities
from .metrics.probabilities import cumulative_distribution
from .metrics.probabilities import get_eigenstate_probabilities
from .metrics.probabilities import statevector_to_probabilities

# solvers
# solver.dynamic
from .solvers.dynamic import evolve_state
# solver.static
from .solvers.static import ground_state
from .solvers.static import find_eigenstate

# lattice_register
from .lattice_register import LatticeRegister

