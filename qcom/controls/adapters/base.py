# qcom/controls/adapters/base.py
from __future__ import annotations

from typing import Protocol, Mapping, runtime_checkable, Any


@runtime_checkable
class ControlAdapter(Protocol):
    """
    Minimal adapter interface for the dynamic solver.

    The solver will:
      1) Sample the required control channels from a TimeSeries at time t.
      2) Call `hamiltonian_at(t, controls)` to get the full H(t).
      3) Exponentiate H(t) (dense or sparse) to advance the state.

    Required
    --------
    • required_channels : tuple[str, ...]
        Channel names this adapter needs (case-sensitive).

    • hamiltonian_at(t, controls) -> <matrix-like>
        Return the full Hamiltonian H(t). Supported return types:
          - dense numpy.ndarray (complex128 recommended),
          - scipy.sparse CSR/CSC matrix,
          - or an object exposing `.to_sparse()` (and optionally `.to_matrix()`).

    Optional
    --------
    • dimension : int
        If provided and nonzero, used to validate the size of the initial state.

    • plot_labels_abs / plot_labels_norm / plot_norm_y_hints
        Cosmetic hints used by plotting presets (if you choose to provide them).
    """

    # ------------------- REQUIRED -------------------
    @property
    def required_channels(self) -> tuple[str, ...]: ...
    def hamiltonian_at(self, t: float, controls: Mapping[str, float]) -> Any: ...

    # ------------------- OPTIONAL -------------------
    @property
    def dimension(self) -> int: ...  # return 0 if unknown

    # Plotting hints (optional; used by visualization presets if present)
    @property
    def plot_labels_abs(self) -> Mapping[str, str]: ...
    @property
    def plot_labels_norm(self) -> Mapping[str, str]: ...
    @property
    def plot_norm_y_hints(self) -> Mapping[str, tuple[float, float]]: ...


def get_required_channels(adapter: object) -> tuple[str, ...]:
    """
    Back-compat helper:
    Prefer `adapter.required_channels`, but accept legacy `expected_channels`
    if found. Returns () if neither exists.
    """
    if hasattr(adapter, "required_channels"):
        rc = getattr(adapter, "required_channels")
        try:
            return tuple(rc)  # type: ignore[arg-type]
        except Exception:
            return ()
    if hasattr(adapter, "expected_channels"):
        ec = getattr(adapter, "expected_channels")
        try:
            return tuple(ec)  # type: ignore[arg-type]
        except Exception:
            return ()
    return ()