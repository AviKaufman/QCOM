# Contributing To QCOM

This document is the working standard for QCOM. Keep changes small, explicit,
and easy to validate.

## Repository Layout

- Library code lives in `src/qcom`.
- Tests live in `tests` and should mirror the package subsystem they cover.
- Examples live in `examples` and should be runnable scripts.
- Tutorials live in `tutorials` and may keep saved outputs when they help the
  lesson. Re-execute notebooks before committing so any outputs are current and
  intentional.
- Development utilities live in `scripts`.
- `README.md` is the user-facing front door and documentation map.
- `CONTRIBUTING.md` is the canonical contributor standard.
- `repo-landscape.md` is the source-grounded architecture map.
- `ROADMAP.md` is the backlog for known issues, maintenance, compatibility,
  and future feature work.

## Docs Ownership

| File | Owns | Should Not Own |
| --- | --- | --- |
| `README.md` | Installation, quick usage, documentation map, and development entrypoints | Detailed architecture, contributor rules, or backlog details |
| `CONTRIBUTING.md` | Naming, comments, Markdown, tests, tutorials, validation, and PR hygiene | Architecture walkthroughs or feature planning |
| `repo-landscape.md` | Package layout, subsystem responsibilities, data flows, visual graphs, and reading order | Roadmap promises or contribution policy |
| `ROADMAP.md` | Current state, known issues, maintenance work, compatibility policy, and future features | Install instructions or architectural deep dives |

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

### Function Names

Function names should say what the caller gets or what action happens. Use full
words, not abbreviations. Prefer `compute_cumulative_distribution` over
`compute_cd`, `comp_cd`, or `cdf_calc`.

Use this decision tree for new functions:

- Use `compute_<quantity>` for derived numerical or analytical quantities.
  Examples: `compute_shannon_entropy`, `compute_mutual_information`,
  `compute_reduced_density_matrix`, `compute_cumulative_distribution`.
- Use `<source>_to_<target>` for pure conversions where the output type is the
  main point. Examples: `statevector_to_probabilities`,
  `normalize_to_probabilities`.
- Use `build_<model>` for constructing model objects from domain inputs.
  Examples: `build_ising`, `build_rydberg`.
- Use `parse_<format>` for reading external formats into QCOM data structures.
  Examples: `parse_aquila_json`, `parse_parquet`, `parse_text`.
- Use `save_<target>` or `save_<object>_to_<format>` for writing data.
  Examples: `save_text`, `save_parquet`.
- Use imperative verbs for operations that alter, sample, combine, or present
  data. Examples: `sample_counts`, `combine_bitstring_datasets`,
  `truncate_probabilities`, `apply_readout_error`,
  `print_most_probable_bitstrings`.
- Use `get_<thing>` only for lightweight accessors or compatibility helpers.
  Avoid `get_` for expensive numerical work.
- Use `is_<condition>`, `has_<thing>`, or `needs_<thing>` for boolean predicates.

For the cumulative-distribution example, new code should use
`compute_cumulative_distribution` if it derives the distribution from data.
Existing compatibility aliases such as `cumulative_distribution` are
grandfathered for API stability. Do not remove stable public functions without a
deprecation plan.

### Variable Names

- Use nouns that describe the data and units when helpful:
  `probabilities`, `counts`, `n_sites`, `ground_rate`, `time_grid`.
- Avoid one-letter names except in short mathematical scopes where the meaning
  is conventional and local, such as `i`, `j`, `n`, or `H`.
- Include units in names when two unit systems could be confused:
  `times_seconds`, `positions_m`, `omega_rad_s`.
- Prefer domain terms over generic containers. Use `counts` instead of `data`
  when the values are counts; use `probabilities` when values are probabilities.
- Avoid container words such as `dict` and `data` in public names unless the
  concept is genuinely format-agnostic. Prefer
  `marginalize_bitstring_distribution` over `part_dict`,
  `apply_readout_error` over `introduce_error`, and `save_parquet` over
  `save_dict_to_parquet`.
- Use plural names for collections and singular names for one item:
  `bitstrings` versus `bitstring`.

### Class And Type Names

- Use `PascalCase` nouns for classes and protocols.
  Examples: `LatticeRegister`, `RydbergHamiltonian`, `ControlAdapter`.
- Use the suffix `Result` for immutable calculation outputs.
  Examples: `SpectrumResult`, `EvolutionResult`.
- Use the suffix `Data` for typed measurement or probability containers.
  Examples: `CountsData`, `ProbabilityData`.
- Use the suffix `Params` for immutable parameter bundles.
  Examples: `RydbergParams`, `IsingParams`.

### Abbreviations

- Avoid abbreviations in public names unless the abbreviation is a standard
  domain term used by the audience.
- Allowed domain abbreviations include `io`, `json`, `csv`, `cdf` in prose,
  `rdm`, `vnee`, and common physics symbols inside local math-heavy code.
- Prefer full words in APIs even when a shorter acronym is familiar:
  `compute_cumulative_distribution` over `compute_cdf`.
- If an abbreviation appears in a public name, document it in the docstring.
- Compatibility aliases should emit `DeprecationWarning` with the preferred
  replacement name. Keep aliases grouped, documented, and tested with
  `pytest.warns`.

## Compatibility And Deprecation

- Prefer the new public name in fresh code, examples, tutorials, and tests.
- Keep a compatibility alias only when it protects existing callers.
- An alias should remain available for at least one minor release after the
  preferred name is introduced, unless a later release note explicitly extends
  the window.
- Deprecation warnings should name both the old symbol and the preferred
  replacement, and they should fire once per call path.
- Each alias needs a focused behavior test and a warning test before release.
- Migration notes for removals belong in release notes and `ROADMAP.md`, not
  scattered through code comments.
- Preserve keyword behavior in wrappers unless a keyword rename is itself part
  of the migration plan.

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
- Keep roadmap and known-issue items in `ROADMAP.md`. If an item is uncertain,
  keep it broad and mark it as future work rather than promising a date.
- Keep `README.md` focused on user-facing orientation. Put contributor rules here.
- Update `repo-landscape.md` only when architecture facts change.
- When `ROADMAP.md` changes, review `README.md`, `CONTRIBUTING.md`, and
  `repo-landscape.md` in the same pass and update any linked docs that are now
  out of sync.

## Tests And Validation

- Add focused pytest coverage for new public behavior.
- Optional dependency tests must either install the needed extra in the relevant
  nox session or skip cleanly.
- Optional dependency coverage belongs in `nox -s test_extras` and should stay
  green whenever optional features or dependency wiring change.
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

## Dead Code Review

- Use `vulture` as a review aid, not as an automatic delete list.
- Expect lazy exports, compatibility aliases, and intentional facade helpers to
  appear as false positives.
- Before removing an apparently unused symbol, trace its package exports,
  compatibility wrappers, tests, notebooks, and docs references.
- If a false positive is expected to remain, keep a short comment or roadmap
  note that explains why.

## Tutorials

- Keep notebook outputs current and intentional in git. Avoid stale outputs from
  failed or exploratory runs.
- Preserve the tutorial teaching style: Markdown should explain the reasoning,
  and code cells should show real usage.
- Validate notebooks with `scripts/validate_tutorials.py`.
- Do not compress tutorials into terse examples when the goal is teaching.

## Pull Request Hygiene

- Keep each change scoped to one intent.
- Separate mechanical formatting from behavior changes when practical.
- Include the validation commands and results in the final report.
- Call out intentional follow-up work instead of hiding known inconsistencies.
