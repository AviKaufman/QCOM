# QCOM

**Quantum Computation (QCOM)** is a Python package developed as part of Avi Kaufman’s 2025 honors thesis in physics. Designed to support the **Meurice Research Group**, QCOM focuses on analyzing thermodynamic properties of quantum systems — particularly those involving neutral atom (Rydberg) platforms.

QCOM enables users to compute exact results for model Hamiltonians, analyze probability distributions from external sources such as DMRG or quantum hardware (e.g., QuEra’s Aquila), and calculate both classical and quantum information measures such as Shannon entropy, von Neumann entropy, and mutual information.

---

## Installation

You can install the latest pre-release of QCOM directly from PyPI:

```bash
pip install --pre QCOM
```

## Confirm Installation

In the python environement you've installed qcom, running the following code:

```python
import qcom
print(dir(qcom))
```

You should see an output like this:

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

## Project Status

QCOM is an active work in progress. New features will be added to meet the evolving needs of the Meurice Group or other researchers. Suggestions, bug reports, and collaborations are welcome.

**Last updated:** March 27, 2025