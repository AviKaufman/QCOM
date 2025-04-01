# QCOM

**Quantum Computation (QCOM)** is a Python package developed as part of Avi Kaufman’s 2025 honors thesis in physics. Designed to support the **Meurice Research Group**, QCOM focuses on analyzing thermodynamic properties of quantum systems — particularly those involving neutral atom (Rydberg) platforms.

QCOM enables users to compute exact results for model Hamiltonians, analyze probability distributions from external sources such as DMRG or quantum hardware (e.g., QuEra’s Aquila), and calculate both classical and quantum information measures such as Shannon entropy, von Neumann entropy, and mutual information.

---

## Installation

You can install the latest release of QCOM directly from PyPI:

```bash
pip install QCOM
```

## Confirm Installation

In the python environement you've installed qcom, running the following code:

```python
import qcom
print(dir(qcom))
```

You should see an output like this:

```text

['ProgressManager', '__author__', '__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__path__', '__spec__', '__version__', 'build_ising_hamiltonian', 'build_ising_hamiltonian_ladder', 'build_rydberg_hamiltonian_chain', 'build_rydberg_hamiltonian_ladder', 'classical_info', 'combine_datasets', 'compute_N_of_p', 'compute_N_of_p_all', 'compute_mutual_information', 'compute_reduced_density_matrix', 'compute_reduced_shannon_entropy', 'compute_shannon_entropy', 'contextmanager', 'create_density_matrix', 'csr_matrix', 'cumulative_distribution', 'data', 'eigsh', 'find_eigenstate', 'get_eigenstate_probabilities', 'hamiltonians', 'identity', 'introduce_error_data', 'io', 'kron', 'normalize_to_probabilities', 'np', 'order_dict', 'os', 'parse_file', 'parse_parq', 'part_dict', 'pd', 'print_most_probable_data', 'quantum_info', 'random', 'sample_data', 'save_data', 'save_dict_to_parquet', 'sys', 'time', 'utils', 'von_neumann_entropy_from_hamiltonian', 'von_neumann_entropy_from_rdm']
```

## Core Capabilities

- Build exact Hamiltonians for:
  - 1D Rydberg atom chains and ladders
  - Quantum Ising models in chain and ladder geometries

- Efficiently compute:
  - Ground states and eigenstate properties
  - Von Neumann entanglement entropy from a Hamiltonian or reduced density matrix
  - Shannon entropy and mutual information from classical distributions

- Parse, normalize, and sample binary data from experimental or simulation sources

- Apply and study noise models (bit-flip errors) on binary datasets

- Save and load results in standard formats (`.txt`, `.parquet`)

- Monitor long computations using a flexible `ProgressManager`

---

## Example Use Cases

- Construct a ladder Rydberg Hamiltonian and compute its ground state entropy  
- Parse a binary probability dataset from an experiment and calculate classical mutual information  
- Simulate the effects of readout error on a quantum distribution  
- Combine or sample from large bitstring datasets for postprocessing

---

## Tutorials



---

## Testing

The unit tests included in this repository are designed to verify that core functions behave as expected under typical use cases. While they provide useful coverage of the package’s functionality, they are not exhaustive and do not guarantee the absence of logical errors in every edge case. Continued validation, peer review, and scientific scrutiny are encouraged to ensure the accuracy and reliability of the results produced by this package.

---

## Project Status

QCOM is an active work in progress. New features will be added to meet the evolving needs of the Meurice Group or other researchers. Suggestions, bug reports, and collaborations are welcome.

**Last updated:** March 31, 2025