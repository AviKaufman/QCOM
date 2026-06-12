"""Nox sessions for QCOM development."""

from __future__ import annotations

import nox

nox.options.default_venv_backend = "uv|virtualenv"
nox.options.sessions = ["lint", "typecheck", "test", "build"]


@nox.session(python="3.12")
def lint(session: nox.Session) -> None:
    session.install("ruff")
    session.run("ruff", "format", "--check", "src", "tests", "examples", "scripts", "noxfile.py")
    session.run("ruff", "check", "src", "tests", "examples", "scripts", "noxfile.py")


@nox.session(python="3.12")
def typecheck(session: nox.Session) -> None:
    session.install("-e", ".[dev,parquet,viz]")
    session.run("mypy")


@nox.session(python=["3.10", "3.11", "3.12", "3.13"])
def test(session: nox.Session) -> None:
    session.install("-e", ".[dev,parquet,viz]")
    session.run("pytest")


@nox.session(python="3.12")
def test_extras(session: nox.Session) -> None:
    session.install("-e", ".[all,dev]")
    session.run("pytest")


@nox.session(python="3.12")
def tutorials(session: nox.Session) -> None:
    session.install("-e", ".[dev,parquet,viz]", "nbformat", "nbclient", "ipykernel")
    session.run("python", "scripts/validate_tutorials.py")


@nox.session(python="3.12")
def build(session: nox.Session) -> None:
    session.install("build")
    session.run("python", "-m", "build")
