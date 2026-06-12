import types
import sys
import numpy as np
import pytest

from qcom.data.noise import apply_readout_error, introduce_error, m3_mitigate_counts_from_rates


def test_apply_readout_error_no_noise_identity():
    """With zero error rates, output must equal input exactly."""
    counts = {"00": 10, "01": 5, "10": 7, "11": 0}
    counts_no_11 = {"00": 10, "01": 5, "10": 7}  # also test missing keys
    out = apply_readout_error(counts, ground_rate=0.0, excited_rate=0.0, seed=123)
    assert out == counts_no_11  # exact identity


def test_apply_readout_error_full_flip_complement():
    """With unit error rates, every bit flips (0->1 and 1->0)."""
    counts = {"00": 3, "01": 2, "10": 4, "11": 1}
    out = apply_readout_error(counts, ground_rate=1.0, excited_rate=1.0, seed=42)
    # Expect full complement of each key with the same counts
    expected = {"11": 3, "10": 2, "01": 4, "00": 1}
    assert out == expected


def test_apply_readout_error_seed_reproducibility():
    """Same seed gives the same result; a different seed likely differs."""
    counts = {"010": 100}
    out1 = apply_readout_error(counts, ground_rate=0.3, excited_rate=0.6, seed=7)
    out2 = apply_readout_error(counts, ground_rate=0.3, excited_rate=0.6, seed=7)
    out3 = apply_readout_error(counts, ground_rate=0.3, excited_rate=0.6, seed=8)

    assert out1 == out2
    assert out3 != out1


def test_apply_readout_error_probability_mass_conserved():
    """Total number of shots must be conserved."""
    counts = {"000": 12, "111": 8}
    total_in = sum(counts.values())
    out = apply_readout_error(counts, ground_rate=0.2, excited_rate=0.2, seed=123)
    assert sum(out.values()) == total_in


def test_apply_readout_error_per_site_sequence():
    """Per-site sequences should be respected and length-checked."""
    counts = {"010": 50, "101": 50}  # N = 3
    # With p01 all zeros, no 0->1 flips should occur; with p10 all ones, every '1' flips to '0'
    out = apply_readout_error(
        counts,
        ground_rate=[0.0, 0.0, 0.0],
        excited_rate=[1.0, 1.0, 1.0],
        seed=1,
    )
    for k in out.keys():
        assert set(k) <= {"0"}

    with pytest.raises(ValueError):
        _ = apply_readout_error(counts, ground_rate=[0.1, 0.2], excited_rate=0.1, seed=1)


def test_apply_readout_error_per_site_mapping():
    """Sparse mapping of indices should be supported (others default to 0)."""
    counts = {"00": 100}
    out = apply_readout_error(counts, ground_rate={1: 1.0}, excited_rate={1: 1.0}, seed=123)
    assert set(out.keys()) == {"01"}
    assert sum(out.values()) == 100


def test_apply_readout_error_invalid_rates():
    counts = {"0": 1}
    with pytest.raises(ValueError):
        _ = apply_readout_error(counts, ground_rate=-0.1, excited_rate=0.0)
    with pytest.raises(ValueError):
        _ = apply_readout_error(counts, ground_rate=0.0, excited_rate=1.1)


def test_introduce_error_compatibility_alias():
    counts = {"0": 2}
    with pytest.warns(DeprecationWarning, match="apply_readout_error"):
        alias_counts = introduce_error(counts, ground_rate=0.0, excited_rate=0.0, seed=1)
    assert alias_counts == apply_readout_error(
        counts,
        ground_rate=0.0,
        excited_rate=0.0,
        seed=1,
    )


class _FakeM3Mitigation:
    """Minimal stand-in for mthree.M3Mitigation."""

    def cals_from_matrices(self, matrices):
        _ = [np.asarray(m) for m in matrices]

    def apply_correction(self, int_counts, meas_qubits):
        _ = meas_qubits
        total = sum(int_counts.values())
        if total == 0:
            return {}
        probs = {k: v / total for k, v in int_counts.items()}

        # Simple smoothing toward uniform; clip negatives (shouldn't happen) and renormalize.
        keys = sorted(probs.keys())
        uniform = 1.0 / len(keys)
        out = {k: 0.8 * probs[k] + 0.2 * uniform for k in keys}
        s = sum(out.values())
        out = {k: v / s for k, v in out.items()}
        return out


def _install_fake_mthree(monkeypatch):
    """Insert a fake 'mthree' module in sys.modules with the above M3Mitigation."""
    fake_mod = types.ModuleType("mthree")
    fake_mod.M3Mitigation = _FakeM3Mitigation
    monkeypatch.setitem(sys.modules, "mthree", fake_mod)
    return fake_mod


def test_m3_mitigate_counts_empty_returns_empty(monkeypatch):
    _install_fake_mthree(monkeypatch)
    assert m3_mitigate_counts_from_rates({}) == {}


def test_m3_mitigate_counts_per_site_rates_and_custom_qubits(monkeypatch):
    _install_fake_mthree(monkeypatch)

    counts = {"000": 100, "111": 50}
    p01 = [0.0, 0.1, 0.2]
    p10 = [0.3, 0.2, 0.1]
    qubits = [2, 0, 1]

    out = m3_mitigate_counts_from_rates(
        counts,
        ground_rate=p01,
        excited_rate=p10,
        qubits=qubits,
    )

    assert set(out.keys()) == set(counts.keys())
    assert pytest.approx(sum(out.values()), rel=1e-12, abs=1e-12) == 1.0
    assert all(v >= 0.0 for v in out.values())


def test_m3_mitigate_counts_unequal_bit_lengths_error(monkeypatch):
    _install_fake_mthree(monkeypatch)
    counts = {"0": 5, "10": 7}  # inconsistent bit lengths
    with pytest.raises(ValueError):
        _ = m3_mitigate_counts_from_rates(counts, ground_rate=0.01, excited_rate=0.08)


def test_m3_mitigate_counts_invalid_rates_raise(monkeypatch):
    _install_fake_mthree(monkeypatch)
    counts = {"0": 5, "1": 7}
    with pytest.raises(ValueError):
        _ = m3_mitigate_counts_from_rates(counts, ground_rate=-0.1, excited_rate=0.0)
    with pytest.raises(ValueError):
        _ = m3_mitigate_counts_from_rates(counts, ground_rate=0.0, excited_rate=1.5)
