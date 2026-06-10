"""Execute QCOM tutorial notebooks without saving generated outputs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
TUTORIALS = ROOT / "tutorials"


def _tutorial_paths(selected: list[str] | None) -> list[Path]:
    if selected:
        return [(TUTORIALS / name).with_suffix(".ipynb") for name in selected]
    return sorted(TUTORIALS.glob("tutorial_*.ipynb"))


def _assert_clear_outputs(path: Path) -> None:
    nb = nbformat.read(path, as_version=4)
    dirty_cells = [
        index
        for index, cell in enumerate(nb.cells)
        if cell.cell_type == "code"
        and (cell.get("outputs") or cell.get("execution_count") is not None)
    ]
    if dirty_cells:
        raise AssertionError(
            f"{path.name} has saved outputs/execution counts in cells {dirty_cells}."
        )


def _execute(path: Path, *, timeout: int) -> None:
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(TUTORIALS)}},
    )
    client.execute()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebooks", nargs="*", help="Optional notebook basenames to execute.")
    parser.add_argument("--timeout", type=int, default=300, help="Per-cell timeout in seconds.")
    parser.add_argument(
        "--skip-clear-output-check",
        action="store_true",
        help="Execute notebooks even if saved outputs are present.",
    )
    args = parser.parse_args()

    os.environ.setdefault("MPLBACKEND", "Agg")

    paths = _tutorial_paths(args.notebooks)
    if not paths:
        raise SystemExit("No tutorials found.")

    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        if not args.skip_clear_output_check:
            _assert_clear_outputs(path)
        print(f"Executing {path.relative_to(ROOT)}")
        _execute(path, timeout=args.timeout)

    print(f"Executed {len(paths)} tutorial notebook(s).")


if __name__ == "__main__":
    main()
