# tests/test_rydberg_adapter.py
import numpy as np
import scipy.sparse as sp
import pytest

from qcom.controls.adapters.rydberg import RydbergAdapter

# --- Minimal register stub (adapter only needs .sites length for a cached hint) ---
class _DummyRegister:
    def __init__(self, n_sites=2):
        self.sites = [(float(i), 0.0, 0.0) for i in range(n_sites)]


# --- Dummy Hamiltonian containers used by monkeypatched build_rydberg ---

class _DummyH_Sparse:
    """Mocks a Hamiltonian object exposing .to_sparse()."""
    def __init__(self, Omega, Delta, Phi):
        # keep inputs for assertions
        self.Omega = Omega
        self.Delta = Delta
        self.Phi = Phi
        # tiny 2x2 CSR just to have a valid sparse matrix
        self._csr = sp.csr_matrix(np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128))

    def to_sparse(self):
        return self._csr


class _DummyH_Dense:
    """Mocks a Hamiltonian object exposing .to_matrix()."""
    def __init__(self, Omega, Delta, Phi):
        self.Omega = Omega
        self.Delta = Delta
        self.Phi = Phi
        self._mat = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)

    def to_matrix(self):
        return self._mat


# ------------------------------------------------------------------------------------
# Basic properties
# ------------------------------------------------------------------------------------

def test_required_channels_and_dimension_hint():
    reg = _DummyRegister(n_sites=2)
    ad_nohint = RydbergAdapter(register=reg, C6=1.23)
    assert ad_nohint.required_channels == ("Omega", "Delta", "Phi")
    assert ad_nohint.dimension == 0  # default: no hint provided

    ad_hint = RydbergAdapter(register=reg, C6=1.23, hilbert_dim=4)
    assert ad_hint.dimension == 4


# ------------------------------------------------------------------------------------
# Absolute mode (no per-site scaling arrays provided) → broadcast if arrays exist
# ------------------------------------------------------------------------------------

def test_hamiltonian_at_absolute_mode_monkeypatched_sparse(monkeypatch):
    captured = {}

    def fake_build_rydberg(register, C6, Omega, Delta, Phi):
        # capture parameters for assertions
        captured["register"] = register
        captured["C6"] = C6
        captured["Omega"] = Omega
        captured["Delta"] = Delta
        captured["Phi"] = Phi
        return _DummyH_Sparse(Omega, Delta, Phi)

    # Patch the symbol used inside the adapter module
    monkeypatch.setattr("qcom.controls.adapters.rydberg.build_rydberg", fake_build_rydberg)

    reg = _DummyRegister(n_sites=2)
    adapter = RydbergAdapter(register=reg, C6=2.5)  # no omega_max / delta_span → absolute mode

    controls = {"Omega": 2.0e6, "Delta": -1.0e6, "Phi": 0.3}
    H = adapter.hamiltonian_at(1.0e-6, controls)

    # Should prefer sparse path when .to_sparse() exists
    assert sp.isspmatrix_csr(H) or sp.isspmatrix(H)

    # In absolute mode (no scaling provided), pass scalars through unchanged
    assert np.isscalar(captured["Omega"]) and captured["Omega"] == controls["Omega"]
    assert np.isscalar(captured["Delta"]) and captured["Delta"] == controls["Delta"]
    assert np.isscalar(captured["Phi"]) and captured["Phi"] == controls["Phi"]


# ------------------------------------------------------------------------------------
# Normalized mode (per-site scaling provided) → scale controls elementwise
# ------------------------------------------------------------------------------------

def test_hamiltonian_at_normalized_scaling_with_phi_offset_scalar(monkeypatch):
    captured = {}

    def fake_build_rydberg(register, C6, Omega, Delta, Phi):
        captured["Omega"] = np.asarray(Omega)
        captured["Delta"] = np.asarray(Delta)
        captured["Phi"] = np.asarray(Phi)
        return _DummyH_Sparse(Omega, Delta, Phi)

    monkeypatch.setattr("qcom.controls.adapters.rydberg.build_rydberg", fake_build_rydberg)

    reg = _DummyRegister(n_sites=3)
    omega_max = np.array([1.0e6, 2.0e6, 3.0e6])   # per-site Ω_max
    delta_span = np.array([0.5e6, 0.5e6, 0.5e6])  # per-site Δ scaling
    phi_offset = 0.1

    adapter = RydbergAdapter(
        register=reg,
        C6=3.14,
        omega_max=omega_max,
        delta_span=delta_span,
        phi_offset=phi_offset,
    )

    # Normalized inputs (unitless Ω, Δ)
    controls = {"Omega": 0.5, "Delta": -0.2, "Phi": 0.0}
    _ = adapter.hamiltonian_at(0.0, controls)

    # Expect elementwise scaling + phi_offset addition
    np.testing.assert_allclose(captured["Omega"], 0.5 * omega_max)
    np.testing.assert_allclose(captured["Delta"], -0.2 * delta_span)
    np.testing.assert_allclose(captured["Phi"], np.full(3, 0.0 + phi_offset))


def test_hamiltonian_at_absolute_broadcast_and_phi_offset_array(monkeypatch):
    captured = {}

    def fake_build_rydberg(register, C6, Omega, Delta, Phi):
        captured["Omega"] = np.asarray(Omega)
        captured["Delta"] = np.asarray(Delta)
        captured["Phi"] = np.asarray(Phi)
        return _DummyH_Sparse(Omega, Delta, Phi)

    monkeypatch.setattr("qcom.controls.adapters.rydberg.build_rydberg", fake_build_rydberg)

    reg = _DummyRegister(n_sites=2)
    # Provide per-site arrays but we want absolute mode:
    # normalized=True is chosen if omega_max or delta_span is provided.
    # To test absolute broadcast, pass arrays as "hooks" but set normalized=False via code path.
    # Since adapter decides normalized = (omega_max or delta_span is not None),
    # we simulate absolute broadcast by not providing them and instead relying on _maybe_broadcast.
    # So here we DO NOT pass omega_max/delta_span → absolute mode.
    phi_offset = np.array([0.1, -0.2])

    adapter = RydbergAdapter(
        register=reg,
        C6=1.0,
        phi_offset=phi_offset,
    )

    controls = {"Omega": 2.0e6, "Delta": 1.0e6, "Phi": 0.05}
    _ = adapter.hamiltonian_at(0.0, controls)

    # With no omega_max/delta_span, scalars should pass through unchanged
    # Phi should add the per-site offset → becomes a length-2 array
    assert captured["Omega"].shape == ()  # scalar
    assert captured["Delta"].shape == ()  # scalar
    np.testing.assert_allclose(captured["Phi"], controls["Phi"] + phi_offset)


# ------------------------------------------------------------------------------------
# Dense fallback when no .to_sparse(), but .to_matrix() exists
# ------------------------------------------------------------------------------------

def test_dense_fallback_when_no_sparse(monkeypatch):
    returned = {}

    def fake_build_rydberg(register, C6, Omega, Delta, Phi):
        obj = _DummyH_Dense(Omega, Delta, Phi)
        returned["obj"] = obj
        return obj

    monkeypatch.setattr("qcom.controls.adapters.rydberg.build_rydberg", fake_build_rydberg)

    reg = _DummyRegister(n_sites=2)
    adapter = RydbergAdapter(register=reg, C6=1.0)

    H = adapter.hamiltonian_at(0.0, {"Omega": 1.0, "Delta": 0.0, "Phi": 0.0})
    assert isinstance(H, np.ndarray)
    np.testing.assert_array_equal(H, returned["obj"]._mat)


# ------------------------------------------------------------------------------------
# Cosmetic plotting hints (existence and basic structure)
# ------------------------------------------------------------------------------------

def test_plotting_hints_shapes():
    reg = _DummyRegister(n_sites=2)
    adapter = RydbergAdapter(register=reg, C6=1.0)

    # Optional properties should be present and mappable
    lab_abs = adapter.plot_labels_abs
    lab_norm = adapter.plot_labels_norm
    y_hints = adapter.plot_norm_y_hints

    assert isinstance(lab_abs, dict) and "omega" in lab_abs and "delta" in lab_abs and "phi" in lab_abs
    assert isinstance(lab_norm, dict) and "omega" in lab_norm and "delta" in lab_norm and "phi" in lab_norm
    assert isinstance(y_hints, dict) and "omega" in y_hints and "delta" in y_hints
    assert isinstance(y_hints["omega"], tuple) and len(y_hints["omega"]) == 2