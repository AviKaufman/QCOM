# tests/test_noise.py
import types
import sys
import numpy as np
import pytest

from qcom.data.noise import introduce_error, m3_mitigate_counts_from_rates


# ------------------------------------------------------------------------------------
# introduce_error tests
# ------------------------------------------------------------------------------------

def test_introduce_error_no_noise_identity():
    """With zero error rates, output must equal input exactly."""
    counts = {"00": 10, "01": 5, "10": 7, "11": 0}
    counts_no_11 = {"00": 10, "01": 5, "10": 7}  # also test missing keys
    out = introduce_error(counts, ground_rate=0.0, excited_rate=0.0, seed=123)
    print(out)
    assert out == counts_no_11  # exact identity


def test_introduce_error_full_flip_complement():
    """With unit error rates, every bit flips (0->1 and 1->0)."""
    counts = {"00": 3, "01": 2, "10": 4, "11": 1}
    out = introduce_error(counts, ground_rate=1.0, excited_rate=1.0, seed=42)
    # Expect full complement of each key with the same counts
    expected = {"11": 3, "10": 2, "01": 4, "00": 1}
    assert out == expected


def test_introduce_error_seed_reproducibility():
    """Same seed → same result; different seed → likely different."""
    counts = {"010": 100}
    out1 = introduce_error(counts, ground_rate=0.3, excited_rate=0.6, seed=7)
    out2 = introduce_error(counts, ground_rate=0.3, excited_rate=0.6, seed=7)
    out3 = introduce_error(counts, ground_rate=0.3, excited_rate=0.6, seed=8)

    assert out1 == out2
    # It's possible (though unlikely) that out3 equals out1 for this small case,
    # but statistically it should differ. If flakiness is a concern, relax to
    # a weaker assertion (e.g., same totals, or skip this check).
    assert out3 != out1


def test_introduce_error_probability_mass_conserved():
    """Total number of shots must be conserved."""
    counts = {"000": 12, "111": 8}
    total_in = sum(counts.values())
    out = introduce_error(counts, ground_rate=0.2, excited_rate=0.2, seed=123)
    assert sum(out.values()) == total_in


def test_introduce_error_per_site_sequence():
    """Per-site sequences should be respected and length-checked."""
    counts = {"010": 50, "101": 50}  # N = 3
    # With p01 all zeros, no 0->1 flips should occur; with p10 all ones, every '1' flips to '0'
    out = introduce_error(counts, ground_rate=[0.0, 0.0, 0.0], excited_rate=[1.0, 1.0, 1.0], seed=1)
    # All ones flip to zeros → all outputs must consist solely of '0's
    for k in out.keys():
        assert set(k) <= {"0"}

    # Mismatched-length sequences should raise
    with pytest.raises(ValueError):
        _ = introduce_error(counts, ground_rate=[0.1, 0.2], excited_rate=0.1, seed=1)


def test_introduce_error_per_site_mapping():
    """Sparse mapping of indices should be supported (others default to 0)."""
    counts = {"00": 100}
    # Only flip the LSB with some probability; MSB stays with default=0 → never flips.
    out = introduce_error(counts, ground_rate={1: 1.0}, excited_rate={1: 1.0}, seed=123)
    # From "00": LSB flips 0->1 always; MSB stays 0 (never flips) → "01" only
    assert set(out.keys()) == {"01"}
    assert sum(out.values()) == 100


def test_introduce_error_invalid_rates():
    counts = {"0": 1}
    with pytest.raises(ValueError):
        _ = introduce_error(counts, ground_rate=-0.1, excited_rate=0.0)
    with pytest.raises(ValueError):
        _ = introduce_error(counts, ground_rate=0.0, excited_rate=1.1)


# ------------------------------------------------------------------------------------
# m3_mitigate_counts_from_rates tests (with a fake mthree)
# ------------------------------------------------------------------------------------

class _FakeM3Mitigation:
    """Minimal stand-in for mthree.M3Mitigation capturing input matrices and qubits."""
    def __init__(self):
        self.mats = None
        self.loaded = False
        self.applied_qubits = None

    def cals_from_matrices(self, matrices):
        # Matrices is a list of 2x2 arrays per qubit
        self.mats = [np.asarray(m) for m in matrices]
        self.loaded = True

    def apply_correction(self, int_counts, meas_qubits):
        # Record qubits and return a simple deterministic "mitigated" distribution:
        # Convert counts to normalized probabilities and slightly reweight toward uniform,
        # then renormalize to 1.
        self.applied_qubits = list(meas_qubits)
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

    counts = {"000": 100, "111": 50}  # 3 qubits
    # Per-site rates
    p01 = [0.0, 0.1, 0.2]
    p10 = [0.3, 0.2, 0.1]
    qubits = [2, 0, 1]  # custom order

    out = m3_mitigate_counts_from_rates(
        counts,
        ground_rate=p01,
        excited_rate=p10,
        qubits=qubits,
    )

    # Output should be normalized quasi-probabilities with same key set
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