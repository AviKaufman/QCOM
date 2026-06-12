# QCOM Roadmap

This file is the single backlog source for QCOM. Use [README.md](README.md)
for install and first usage, [CONTRIBUTING.md](CONTRIBUTING.md) for contributor
standards, and [repo-landscape.md](repo-landscape.md) for the architecture map.
When this file changes, review the other Markdown files in the same pass and
update any linked docs that are now out of sync.

## Current State

- QCOM uses a `src/qcom` package layout with tests, examples, and tutorial
  notebooks at the repository root.
- The public facade in `src/qcom/__init__.py` lazily exposes the main geometry,
  Hamiltonian, solver, metrics, data, I/O, and result-container APIs.
- The physics layer currently supports Rydberg and transverse-field Ising
  Hamiltonian workflows.
- Static and dynamic solvers cover small dense/sparse workflows and
  time-dependent evolution through control adapters.
- Data, I/O, metrics, and visualization helpers support measurement-style
  bitstring workflows.
- Compatibility aliases are retained and emit `DeprecationWarning` through
  `src/qcom/_internal/deprecations.py`.
- `mypy` now covers the full `src/qcom` package instead of a hand-picked subset.
- Control adapters are explicit: the registry remains available, but no default
  adapters are auto-registered.
- Optional dependency coverage now runs in CI through `nox -s test_extras`.
- Tutorial notebooks currently execute cleanly under
  `scripts/validate_tutorials.py`.
- Deprecated aliases follow the documented release policy in
  [CONTRIBUTING.md](CONTRIBUTING.md), so old names remain available only as
  retained compatibility shims.
- Standard validation is organized through `noxfile.py`, with extra tutorial
  validation through `scripts/validate_tutorials.py`.

## Immediate Fixes

### Now

- Keep the preferred API names dominant in README examples, tutorials, examples,
  and repo-owned tests.
- Preserve output-free tutorial notebooks after API or documentation changes.

### Next

- Add targeted tests for any adapter-registry behavior that remains public.
- Audit compatibility wrappers before release so every alias warns once per call
  path and forwards keyword arguments predictably.
- Document any non-obvious migration guidance for deprecated names in one place
  rather than scattering notes across examples.

## Repo-Wide Maintenance

### Now

- Keep Markdown ownership clean: README for orientation, CONTRIBUTING for
  standards, repo-landscape for architecture, and ROADMAP for backlog.
- Avoid broad refactors while compatibility aliases are being settled.
- Treat lazy exports and compatibility aliases as intentional when reviewing
  dead-code tooling output.

### Next

- Keep `mypy` coverage on the full `src/qcom` package and consider stricter
  checking only after the current baseline stays green in CI.
- Add release hygiene around build checks, package metadata review, and release
  notes.
- Consider a lightweight dead-code or vulture review process with documented
  false positives for lazy package exports.

### Later

- Introduce more automated documentation checks if the Markdown set grows.
- Add a changelog once release cadence and compatibility policy are explicit.

## API And Compatibility

### Now

- Keep compatibility aliases retained-but-deprecated until the release policy is
  written.
- Keep public names explicit, descriptive, and aligned with
  [CONTRIBUTING.md](CONTRIBUTING.md).
- Avoid renaming public APIs without aliases, warnings, tests, and migration
  notes.

### Next

- Review package `__all__` and lazy facade exports as part of every API change.
- Add focused tests for public re-exports when new subsystems become user-facing.
- Decide whether compatibility aliases should remain in subsystem modules only
  or also be exposed through the top-level facade.

### Later

- Publish a compatibility matrix once multiple versions need to be supported.
- Consider a formal deprecation decorator if alias handling grows beyond simple
  wrappers.

## Documentation And Tutorials

### Now

- Keep tutorials output-free in git and validate them after notebook changes.
- Update docs in the same change as behavior, public API, workflow, or tutorial
  edits.
- Keep old API names out of teaching material except in explicit compatibility
  notes or alias tests.

### Next

- Add short migration notes for deprecated aliases after the release policy is
  decided.
- Expand the docs/site only after the current Markdown ownership remains stable.
- Keep examples runnable and aligned with preferred API names.

### Later

- Consider a generated API reference if public surface area continues to grow.
- Add deeper narrative docs for common research workflows once the underlying
  APIs are stable.

## Feature Roadmap

### Next

- Add more controls and adapter examples through explicit adapter factories and
  registry registration examples.
- Improve data workflows around experiment ingestion, normalization, sampling,
  and readout mitigation.
- Add richer plotting presets for common lattice, control, and distribution
  views.

### Later

- Add additional Hamiltonian families and lattice models where the API can stay
  consistent with the current builder pattern.
- Add parameter-sweep utilities for larger optimization and analysis workloads.
- Explore larger-system solver strategies after the small-system dense/sparse
  interfaces are stable.
- Expand file-format support for experiment data if concrete workflows require
  it.

## Known Issues

- Optional dependencies (`pyarrow`, `matplotlib`, and `mthree`) are intentionally
  isolated, and CI now exercises them through `nox -s test_extras`, but
  release checks should still validate the real environment before shipping.
- Dead-code tools may flag lazy exports, compatibility aliases, or retained
  adapter helpers unless those findings are reviewed with project context.

## Validation Gates

Run the standard validation before broad repo work is considered complete:

```bash
nox -s lint typecheck test build
```

Run tutorial validation before merging notebook or tutorial-reference changes:

```bash
nox -s tutorials
```

Run optional-dependency coverage before release or optional-feature work:

```bash
nox -s test_extras
```

Useful direct equivalents when `nox` or a configured Python version is not
available:

```bash
uv run --with ruff ruff format --check src tests examples scripts noxfile.py
uv run --with ruff ruff check src tests examples scripts noxfile.py
uv run --with pytest --with matplotlib --with pyarrow pytest
uv run --with nbformat --with nbclient --with ipykernel --with matplotlib --with pyarrow python scripts/validate_tutorials.py
```
