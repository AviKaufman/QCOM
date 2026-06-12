"""Execute QCOM tutorial notebooks and verify they still run cleanly."""

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
    args = parser.parse_args()

    os.environ.setdefault("MPLBACKEND", "Agg")

    paths = _tutorial_paths(args.notebooks)
    if not paths:
        raise SystemExit("No tutorials found.")

    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        print(f"Executing {path.relative_to(ROOT)}")
        _execute(path, timeout=args.timeout)

    print(f"Executed {len(paths)} tutorial notebook(s).")


if __name__ == "__main__":
    main()
