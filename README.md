# QCOM

[![PyPI version](https://img.shields.io/pypi/v/QCOM.svg)](https://pypi.org/project/QCOM/)
[![Python versions](https://img.shields.io/pypi/pyversions/QCOM.svg)](https://pypi.org/project/QCOM/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/AviKaufman/QCOM/blob/main/LICENSE.txt)

**Quantum Computation (QCOM)** is a Python package originally developed as part of Avi Kaufman’s 2025 honors thesis, and now maintained as an **ongoing project for quantum systems research**.

QCOM offers a lightweight, extensible toolkit for building model Hamiltonians, evolving states, and analyzing classical/quantum information measures from simulated or experimental bitstring data.

---

## 📦 Installation

```bash
pip install QCOM
```

Upgrade to the latest release:

```bash
pip install --upgrade QCOM
```

For local development, QCOM uses a `src/` layout:

```bash
python -m pip install -e ".[dev,parquet,viz]"
```


⸻

✅ Quick check in python

```python
import qcom
print("QCOM version:", qcom.__version__)
```


⸻

## ✨ Core Capabilities

- **Hamiltonians**
  - Exact builders for **Rydberg** systems (chains/ladders)
  - Working **transverse-field Ising** Hamiltonians with dense, sparse, and matvec backends

- **Solvers**
  - *Static*: thin-spectrum eigen solve, ground-state utilities, dense full spectrum for small systems
  - *Dynamic*: generic time evolution under time-dependent Hamiltonians via matrix exponentials

- **Metrics**
  - *Classical*: Shannon entropy, conditional entropy, mutual information
  - *Quantum*: von Neumann entanglement entropy (from state vectors, density matrices, or Hamiltonians)
  - *Probability tools*: cumulative distributions, N(p) diagnostic, statevector → probabilities

- **Data & Noise**
  - Parse/normalize/sample binary datasets (Plaintext, **Parquet**, **Aquila JSON**)
  - Readout noise models (bit-flip) and optional mitigation via `mthree`

- **I/O**
  - Save/load in plaintext and Parquet
  - Lightweight JSON reader for QuEra **Aquila** results

- **Developer Ergonomics**
  - `ProgressManager` hooks for long tasks
  - Clear conventions (MSB ↔ site 0), endianness controls where relevant

---

## 🚀 Examples

- Build a ladder Rydberg Hamiltonian and compute its ground-state entropy
- Parse measurement data (e.g., from Aquila) and evaluate **mutual information**
- Simulate **readout error** on a probability distribution and apply mitigation
- Sample and **merge** large bitstring datasets for statistical analysis

---

## 📚 Tutorials

Step-by-step notebooks live in the repository:

- **Tutorials directory**:  
  https://github.com/AviKaufman/QCOM/tree/main/tutorials

Suggested order:
1. I/O basics (text, JSON, Parquet)  
2. Lattice registers and geometry  
3. Rydberg Hamiltonians  
4. Static eigen solvers (ground states)  
5. Control time series  
6. Dynamic time evolution  
7. Data utilities (noise, sampling, mitigation)  
8. Metrics (classical + entanglement)

---

## 🧪 Testing

From the project root:

```bash
python -m pip install -e ".[dev,parquet,viz]"
pytest
```

Developer task runner:

```bash
nox -s lint typecheck test build
```
⸻

📂 Example Data

Curated toy datasets for quick experiments:  
- https://github.com/AviKaufman/QCOM/tree/main/example_data

⸻

🗺️ Roadmap
- New Hamiltonians: Heisenberg and additional lattice models
- Parameter sweeps for large optimization workloads
- Tensor-network methods: DMRG / TEBD for large Hilbert spaces
- Expanded I/O readers and richer plotting presets

Community feedback helps shape priorities—feel free to open issues or PRs.

⸻

🤝 Contributing

We welcome contributions of all sizes:
	•	Bug reports, minimal reproductions
	•	Tests and doc improvements
	•	New examples/tutorials
	•	Feature proposals via GitHub Issues

Repo: https://github.com/AviKaufman/QCOM

⸻

📬 Contact

Avi Kaufman — avigkaufman@gmail.com

⸻

Last updated: June 4, 2026

---

## API Notes

`compute_mutual_information` returns the scalar mutual information by default:

```python
from qcom.metrics import compute_mutual_information

probabilities = {"00": 0.5, "11": 0.5}
mi = compute_mutual_information(probabilities, configuration=[0, 1], base=2)
components = compute_mutual_information(
    probabilities,
    configuration=[0, 1],
    base=2,
    return_components=True,
)
print(mi, components.h_a, components.h_b, components.h_ab)
```

Typed measurement containers are available for explicit counts/probability workflows:

```python
from qcom.data import CountsData, normalize_to_probabilities

counts = CountsData({"00": 10, "11": 5})
probabilities = normalize_to_probabilities(counts)
```

Plotting helpers live in `qcom.viz`; existing `.plot()` methods remain as wrappers.
