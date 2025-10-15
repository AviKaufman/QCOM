# tests/test_solvers_dynamic.py
import numpy as np
import pytest

scipy = pytest.importorskip("scipy")  # dynamic uses SciPy expm backends

from qcom.solvers.dynamic import evolve_state, ControlAdapter


# -------------------- Tiny mocks --------------------

class _MockTimeSeries:
    """
    Minimal TimeSeries stub:
      - domain(): (t0, t1)
      - channel_names: iterable of available channel names
      - value_at([t_mid], channels=...): {name: np.array([value])}
    """
    def __init__(self, t0, t1, channel_values):
        """
        channel_values: dict[str, float] of constant values
        """
        self._t0 = float(t0)
        self._t1 = float(t1)
        self._vals = dict(channel_values)
        self.channel_names = tuple(self._vals.keys())

    def domain(self):
        return (self._t0, self._t1)

    def value_at(self, times, channels):
        # Return each requested channel’s constant value as a length-1 array
        out = {}
        for name in channels:
            out[name] = np.array([self._vals[name]], dtype=float)
        return out


class _XDriveAdapterDense(ControlAdapter):
    """
    H = (Omega/2) * sigma_x  (2x2 dense array)
    """
    def __init__(self, omega_required=True, dim=2):
        self._dim = dim
        self._req = ("Omega",) if omega_required else tuple()

    @property
    def required_channels(self):
        return self._req

    @property
    def dimension(self):
        return self._dim

    def hamiltonian_at(self, t, controls):
        Omega = float(controls.get("Omega", 0.0))
        sx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
        return 0.5 * Omega * sx


def _make_sparse_sigma_x():
    import scipy.sparse as sp
    sx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    return sp.csr_matrix(sx)


class _XDriveAdapterSparse(ControlAdapter):
    """
    H = (Omega/2) * sigma_x  (2x2 CSR sparse)
    """
    def __init__(self, dim=2):
        self._dim = dim
        self._sx = _make_sparse_sigma_x()

    @property
    def required_channels(self):
        return ("Omega",)

    @property
    def dimension(self):
        return self._dim

    def hamiltonian_at(self, t, controls):
        import scipy.sparse as sp
        Omega = float(controls.get("Omega", 0.0))
        return (0.5 * Omega) * self._sx  # CSR


# -------------------- Analytic helper --------------------

def analytic_rabi_xdrive(psi0, Omega, T):
    """
    Exact solution for H = (Omega/2) * sigma_x, psi0 in computational basis.
    U = cos(ΩT/2) I - i sin(ΩT/2) σx
    """
    c = np.cos(0.5 * Omega * T)
    s = np.sin(0.5 * Omega * T)
    I = np.eye(2, dtype=np.complex128)
    sx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    U = c * I - 1j * s * sx
    return U @ psi0


# -------------------- Tests --------------------

def test_dense_times_explicit_matches_analytic():
    Omega = 2.0
    t0, t1 = 0.0, 0.3
    ts = _MockTimeSeries(t0, t1, {"Omega": Omega, "Extra": 123.0})
    adapter = _XDriveAdapterDense()

    # One step with explicit times → midpoint rule over whole interval
    psi0 = np.array([1.0, 0.0], dtype=np.complex128)
    times = np.array([t0, t1], dtype=float)

    psi_T, out = evolve_state(
        ts, adapter, psi0, times=times, n_steps=None, record=True, show_progress=False
    )

    psi_ref = analytic_rabi_xdrive(psi0, Omega, T=t1 - t0)
    assert psi_T.shape == (2,)
    assert np.allclose(psi_T, psi_ref, atol=1e-10)
    # recording includes both endpoints
    assert "times" in out and "states" in out
    assert len(out["times"]) == 2 and len(out["states"]) == 2
    # Norm preserved
    assert np.isclose(np.linalg.norm(psi_T), 1.0, atol=1e-12)


def test_dense_nsteps_grid_from_domain():
    Omega = 1.7
    t0, t1 = 0.0, 0.5
    ts = _MockTimeSeries(t0, t1, {"Omega": Omega})
    adapter = _XDriveAdapterDense()

    psi0 = np.array([0.0, 1.0], dtype=np.complex128)
    psi_T, _ = evolve_state(
        ts, adapter, psi0, n_steps=400, times=None, record=False, show_progress=False
    )
    psi_ref = analytic_rabi_xdrive(psi0, Omega, T=t1 - t0)
    # With 400 midpoint steps, error should be very small
    assert np.allclose(psi_T, psi_ref, atol=5e-6)
    assert np.isclose(np.linalg.norm(psi_T), 1.0, atol=1e-12)


def test_recording_has_correct_lengths():
    Omega = 0.3
    t0, t1 = 0.0, 1.0
    ts = _MockTimeSeries(t0, t1, {"Omega": Omega})
    adapter = _XDriveAdapterDense()

    psi0 = np.array([1.0, 0.0], dtype=np.complex128)
    psi_T, out = evolve_state(
        ts, adapter, psi0, n_steps=10, record=True, show_progress=False
    )
    # n_steps=10 → 11 time grid points and 11 states recorded
    assert len(out["times"]) == 11
    assert len(out["states"]) == 11
    assert np.allclose(out["states"][0], psi0)


def test_mutually_exclusive_times_and_nsteps_error():
    Omega = 1.0
    ts = _MockTimeSeries(0.0, 0.1, {"Omega": Omega})
    adapter = _XDriveAdapterDense()
    psi0 = np.array([1.0, 0.0], dtype=np.complex128)
    with pytest.raises(ValueError):
        evolve_state(ts, adapter, psi0, times=[0.0, 0.1], n_steps=1, show_progress=False)


def test_sparse_path_smoke_and_accuracy():
    sp = pytest.importorskip("scipy.sparse")
    # small interval but > 0
    Omega = 0.9
    t0, t1 = 0.0, 0.2
    ts = _MockTimeSeries(t0, t1, {"Omega": Omega})
    adapter = _XDriveAdapterSparse()

    psi0 = np.array([1.0, 0.0], dtype=np.complex128)
    psi_T, _ = evolve_state(
        ts, adapter, psi0, n_steps=64, record=False, show_progress=False
    )
    psi_ref = analytic_rabi_xdrive(psi0, Omega, T=t1 - t0)

    # Sparse path should agree closely with analytic
    assert np.allclose(psi_T, psi_ref, atol=5e-6)
    assert np.isclose(np.linalg.norm(psi_T), 1.0, atol=1e-12)


def test_disable_normalization_still_unitary():
    Omega = 0.4
    t0, t1 = 0.0, 0.7
    ts = _MockTimeSeries(t0, t1, {"Omega": Omega})
    adapter = _XDriveAdapterDense()
    # random normalized initial state
    rng = np.random.default_rng(123)
    psi0 = rng.normal(size=2) + 1j * rng.normal(size=2)
    psi0 = psi0 / np.linalg.norm(psi0)

    psi_T, _ = evolve_state(
        ts, adapter, psi0, n_steps=128, normalize_each_step=False, show_progress=False
    )
    # Unitary evolution → preserve norm to tight tolerance
    assert np.isclose(np.linalg.norm(psi_T), 1.0, atol=1e-10)