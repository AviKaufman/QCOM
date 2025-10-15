"""
qcom.hamiltonians
=================

Package of Hamiltonian builders and small utilities.

Design
------
- Builders consume lightweight inputs (e.g. a LatticeRegister and scalars/arrays)
  and return *Hamiltonian objects*. These objects are lazy and expose
  materializers like `.to_sparse()`, `.to_dense()`, `.to_linear_operator()`, etc.
- Submodules are imported lazily so `import qcom.hamiltonians` stays fast.

Typical usage
-------------
from qcom.hamiltonians import IsingHamiltonian, build_ising

# Either via the convenience function…
H = build_ising(register, J=..., hx=..., hz=...)

# …or via the class API directly
from qcom.hamiltonians import IsingHamiltonian
H = IsingHamiltonian.from_register(register, J=..., hx=..., hz=...)

Public API (re-exported lazily)
-------------------------------
- BaseHamiltonian                         -> .base
- IsingHamiltonian, build_ising           -> .ising
- RydbergHamiltonian, build_rydberg       -> .rydberg
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = [
    "BaseHamiltonian",
    "IsingHamiltonian",
    "RydbergHamiltonian",
    "build_ising",
    "build_rydberg",
]

# -------------------- Type-only imports (no runtime import cost) --------------------
if TYPE_CHECKING:
    from .base import BaseHamiltonian
    from .ising import IsingHamiltonian
    from .rydberg import RydbergHamiltonian


# -------------------- Lazy attribute loader --------------------
def __getattr__(name: str):
    if name in {"BaseHamiltonian"}:
        from . import base as _base
        return getattr(_base, name)

    if name in {"IsingHamiltonian", "build_ising"}:
        from . import ising as _ising
        return getattr(_ising, name)

    if name in {"RydbergHamiltonian", "build_rydberg"}:
        from . import rydberg as _rydberg
        return getattr(_rydberg, name)

    raise AttributeError(f"module 'qcom.hamiltonians' has no attribute {name!r}")


def __dir__():
    # Helps IDEs / tab-completion see the lazily exposed names
    return sorted(set(list(globals().keys()) + __all__))


# -------------------- Convenience wrappers (remain lazy internally) --------------------
def build_ising(*args, **kwargs):
    """
    Convenience builder for the transverse-field Ising family.

    Returns an `IsingHamiltonian` object. Use its `.to_sparse()`,
    `.to_dense()`, or `.to_linear_operator()` methods to materialize a
    specific representation.
    """
    from .ising import build_ising as _impl
    return _impl(*args, **kwargs)


def build_rydberg(*args, **kwargs):
    """
    Convenience builder for Rydberg/AHS-style Hamiltonians.

    Returns a `RydbergHamiltonian` object. Use `.to_sparse()`, `.to_dense()`,
    or `.to_linear_operator()` to materialize.
    """
    from .rydberg import build_rydberg as _impl
    return _impl(*args, **kwargs)