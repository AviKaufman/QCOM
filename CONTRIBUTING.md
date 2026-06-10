# Contributing To QCOM

This document is the working standard for QCOM. Keep changes small, explicit,
and easy to validate.

## Repository Layout

- Library code lives in `src/qcom`.
- Tests live in `tests` and should mirror the package subsystem they cover.
- Examples live in `examples` and should be runnable scripts.
- Tutorials live in `tutorials` and should remain output-free notebooks.
- Development utilities live in `scripts`.
- Architecture notes live in standalone Markdown files such as `repo-landscape.md`.

## Naming

- Python modules and files use lowercase `snake_case.py`.
- Test files use `test_<behavior>.py`.
- Tutorial notebooks use `tutorial_<n>_<topic>.ipynb`.
- Public classes use `PascalCase`.
- Functions, methods, variables, and parameters use `snake_case`.
- Private helpers use a leading underscore, for example `_normalize_state`.
- Constants use `UPPER_SNAKE_CASE`.
- Stable public API names must be intentionally exported through package
  `__init__.py` files and covered by tests.

## Comments And Docstrings

- Comments should explain why something is non-obvious, not restate what each
  line does.
- Avoid large banner comments and stale "future use" comments unless they carry
  active maintenance value.
- Public classes and functions should have docstrings that describe their
  contract, inputs, outputs, and important conventions.
- Private helper docstrings are useful when the helper encodes math, basis
  ordering, optional dependency behavior, or nontrivial validation.
- Remove dead commented-out code instead of leaving it behind.

## Markdown

- Keep Markdown sections short and scannable.
- Prefer relative repo paths in documentation.
- Update docs in the same change as behavior, public API, workflow, or tutorial
  changes.
- Avoid stale roadmap promises. If a roadmap item is uncertain, keep it broad.
- Keep `README.md` focused on user-facing orientation. Put contributor rules here.
- Update `repo-landscape.md` only when architecture facts change.

## Tests And Validation

- Add focused pytest coverage for new public behavior.
- Optional dependency tests must either install the needed extra in the relevant
  nox session or skip cleanly.
- Use result containers explicitly when a test needs to distinguish counts from
  probabilities.
- Run the standard checks before calling repo work done:

```bash
nox -s lint typecheck test build
```

- For tutorial changes, also run:

```bash
nox -s tutorials
```

If `nox` is unavailable in the local environment, use equivalent `uv run --with`
commands and report the exact commands used.

## Tutorials

- Keep notebook outputs and execution counts cleared in git.
- Preserve the tutorial teaching style: Markdown should explain the reasoning,
  and code cells should show real usage.
- Validate notebooks with `scripts/validate_tutorials.py`.
- Do not compress tutorials into terse examples when the goal is teaching.

## Pull Request Hygiene

- Keep each change scoped to one intent.
- Separate mechanical formatting from behavior changes when practical.
- Include the validation commands and results in the final report.
- Call out intentional follow-up work instead of hiding known inconsistencies.
