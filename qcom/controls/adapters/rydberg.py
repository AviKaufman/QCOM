# qcom/controls/adapters/rydberg.py
from __future__ import annotations

from typing import Mapping, Optional
import numpy as np

from qcom.controls.adapters.base import ControlAdapter
from qcom.hamiltonians.rydberg import build_rydberg
from qcom.lattice_register import LatticeRegister


class RydbergAdapter(ControlAdapter):
    """
    Map channel samples {Omega, Delta, Phi} at time t into a full Rydberg Hamiltonian
    using the static builder `build_rydberg(...)`.

    Contract (full-H path only)
    ---------------------------
    • required_channels -> ("Omega", "Delta", "Phi")
    • hamiltonian_at(t, controls) -> matrix-like
        - returns a SciPy sparse matrix (via H.to_sparse()) or a dense ndarray.
    • dimension (optional) -> int
        - if provided and nonzero, used for solver's early size validation.

    Optional site-wise scaling (normalized mode)
    --------------------------------------------
    If `omega_max` and/or `delta_span` are provided, values from a normalized
    TimeSeries can be scaled to per-site arrays at runtime. `phi_offset` may
    be a scalar or per-site array and is simply added.
    """

    # ------------------------------------------ Init ------------------------------------------
    def __init__(
        self,
        *,
        register: LatticeRegister,
        C6: float,
        omega_max: Optional[np.ndarray] = None,       # shape (N,)
        delta_span: Optional[np.ndarray] = None,      # shape (N,)
        phi_offset: Optional[float | np.ndarray] = None,  # scalar or (N,)
        hilbert_dim: Optional[int] = None,
    ):
        self.register = register
        self.C6 = float(C6)
        self._omega_max = omega_max
        self._delta_span = delta_span
        self._phi_offset = phi_offset
        self._dim = int(hilbert_dim) if hilbert_dim is not None else 0

        try:
            self._num_sites = len(self.register.sites)  # adapt if your API differs
        except Exception:
            self._num_sites = None

    # ------------------------------------------ Required by solver ------------------------------------------
    @property
    def required_channels(self) -> tuple[str, ...]:
        # Channels the solver must sample from the TimeSeries
        return ("Omega", "Delta", "Phi")

    @property
    def dimension(self) -> int:
        # Return 0 if unknown (solver will skip early size validation).
        return self._dim

    # ------------------------------------------ Plotting hints (optional) ------------------------------------------
    # These are cosmetic helpers some plotting presets may use if available.
    @property
    def plot_labels_abs(self) -> Mapping[str, str]:
        return {"omega": r"$\Omega$", "delta": r"$\Delta$", "phi": r"$\Phi\,(\mathrm{rad})$"}

    @property
    def plot_labels_norm(self) -> Mapping[str, str]:
        return {
            "omega": r"$\Omega_{\mathrm{env}}$",
            "delta": r"$\Delta_{\mathrm{env}}$",
            "phi":   r"$\Phi_{\mathrm{env}}$",
        }

    @property
    def plot_norm_y_hints(self) -> Mapping[str, tuple[float, float]]:
        # Useful defaults if your TimeSeries is in normalized mode.
        return {"omega": (-0.05, 1.05), "delta": (-1.05, 1.05)}

    # ------------------------------------------ Helpers ------------------------------------------
    def _scale_params_if_needed(
        self,
        controls: Mapping[str, float],
        *,
        normalized: bool = False,
    ) -> dict:
        """
        Turn scalar channel samples into arrays if site-wise scaling is configured,
        or leave as scalars otherwise.
        """
        Omega = controls.get("Omega", 0.0)
        Delta = controls.get("Delta", 0.0)
        Phi   = controls.get("Phi",   0.0)

        def _maybe_broadcast(x, target):
            if target is None:
                return x
            return np.asarray(x) * np.ones_like(target, dtype=np.float64)

        if normalized:
            # In normalized mode, the series typically provides Ω∈[0,1], Δ∈[-1,1]
            if self._omega_max is not None:
                Omega = Omega * self._omega_max
            if self._delta_span is not None:
                Delta = Delta * self._delta_span
        else:
            # Absolute mode; broadcast scalar to site-wise arrays if hooks exist
            Omega = _maybe_broadcast(Omega, self._omega_max)
            Delta = _maybe_broadcast(Delta, self._delta_span)

        if self._phi_offset is not None:
            Phi = Phi + self._phi_offset

        return {"Omega": Omega, "Delta": Delta, "Phi": Phi}

    # ------------------------------------------ Full Hamiltonian path ------------------------------------------
    def hamiltonian_at(self, t: float, controls: Mapping[str, float]):
        """
        Build the full H(t) from current controls using `build_rydberg`.
        Returns a sparse matrix (preferred) if your Hamiltonian object supports `.to_sparse()`.
        """
        # Heuristic: if per-site scaling arrays are provided, assume the series is normalized.
        normalized = (self._omega_max is not None) or (self._delta_span is not None)
        params = self._scale_params_if_needed(controls, normalized=normalized)

        H = build_rydberg(
            register=self.register,
            C6=self.C6,
            Omega=params["Omega"],
            Delta=params["Delta"],
            Phi=params["Phi"],
        )

        # Prefer sparse materialization to save memory; fall back to dense if needed.
        if hasattr(H, "to_sparse"):
            return H.to_sparse()
        if hasattr(H, "to_matrix"):
            return H.to_matrix()
        # Assume builder already returned a matrix-like object.
        return H