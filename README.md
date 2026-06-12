# QCOM

[![PyPI version](https://img.shields.io/pypi/v/QCOM.svg)](https://pypi.org/project/QCOM/)
[![Python versions](https://img.shields.io/pypi/pyversions/QCOM.svg)](https://pypi.org/project/QCOM/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE.txt)

**Quantum Computation (QCOM)** is a lightweight Python toolkit for building
model Hamiltonians, evolving states, and analyzing classical or quantum
information measures from simulated and experimental bitstring data.

## Start Here

Use this README for installation, quick usage, and the main development entry
points. The rest of the Markdown set has one job per file:

| Need | Start With |
| --- | --- |
| Understand the package architecture and data flow | [repo-landscape.md](repo-landscape.md) |
| Follow naming, comments, Markdown, testing, and PR standards | [CONTRIBUTING.md](CONTRIBUTING.md) |
| Pick up known issues, maintenance work, or future features | [ROADMAP.md](ROADMAP.md) |
| Learn workflows step by step | [tutorials](tutorials) |
| Run small script examples | [examples](examples) |

## Installation

```bash
pip install QCOM
```

For local development, install the editable package with the common extras:

```bash
python -m pip install -e ".[dev,parquet,viz]"
```

Quick import check:

```python
import qcom

print("QCOM version:", qcom.__version__)
```

## Core Capabilities

- Hamiltonians: Rydberg and transverse-field Ising builders with dense, sparse,
  and matvec-oriented backends.
- Solvers: static eigen-solvers, ground-state helpers, dense spectra for small
  systems, and dynamic evolution with `expm_multiply`.
- Metrics: Shannon entropy, conditional entropy, mutual information, von
  Neumann entanglement entropy, cumulative distributions, N(p), and
  statevector-to-probability helpers.
- Data utilities: counts/probability containers, normalization, sampling,
  dataset combination, readout-error simulation, and optional `mthree`
  mitigation.
- I/O: plaintext, Parquet, and QuEra Aquila JSON readers.
- Visualization: lattice-register and control-envelope plotting helpers.

QCOM uses the MSB-to-site-0 convention by default. Functions that expose
endianness controls document the alternate little-endian labeling explicitly.

## Quick Usage

```python
from qcom import LatticeRegister, build_ising, ground_state

register = LatticeRegister([(0.0, 0.0, 0.0), (1.0e-6, 0.0, 0.0)])
hamiltonian = build_ising(register, transverse_field=1.0, longitudinal_field=0.2)
energy, state = ground_state(hamiltonian)
```

```python
from qcom.data import CountsData, normalize_to_probabilities, sample_counts

counts = CountsData({"00": 10, "11": 5})
probabilities = normalize_to_probabilities(counts)
sampled_counts = sample_counts(counts.to_dict(), total_count=counts.shots, sample_size=100)
```

```python
from qcom.metrics import compute_mutual_information

probabilities = {"00": 0.5, "11": 0.5}
mutual_information = compute_mutual_information(
    probabilities,
    configuration=[0, 1],
    base=2,
)
```

## Examples And Tutorials

Runnable scripts live in [examples](examples). Step-by-step notebooks live in
[tutorials](tutorials). They are teaching artifacts, so keep outputs current and
intentional when they help the lesson.

Suggested tutorial order:

1. I/O basics
2. Lattice registers and geometry
3. Rydberg Hamiltonians
4. Static eigen-solvers
5. Control time series
6. Dynamic time evolution
7. Data utilities
8. Metrics

Validate tutorials after notebook changes:

```bash
nox -s tutorials
```

## Development

Contributor standards live in [CONTRIBUTING.md](CONTRIBUTING.md). Run the
standard checks from the repository root before calling repo work done:

```bash
nox -s lint typecheck test build
```

Useful focused gates:

```bash
nox -s tutorials
nox -s test_extras
```

The architecture map in [repo-landscape.md](repo-landscape.md) is the fastest
way to orient to the package layout before making subsystem changes. Optional
dependency changes should be checked with `nox -s test_extras`.

## Data

Curated toy datasets for experiments live in [example_data](example_data).

## Roadmap

See [ROADMAP.md](ROADMAP.md) for the current backlog, known issues, maintenance
priorities, compatibility policy work, and future feature direction.

## Contact

Avi Kaufman, avigkaufman@gmail.com
