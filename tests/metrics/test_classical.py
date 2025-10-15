# tests/test_metrics_classical.py
import math
import numpy as np
import pytest

from qcom.metrics.classical import (
    compute_shannon_entropy,
    compute_reduced_shannon_entropy,
    compute_mutual_information,
    compute_conditional_entropy,
)


# -------------------- Fixtures / simple helpers --------------------

def _isclose(a, b, rtol=1e-9, atol=1e-12):
    return np.isclose(a, b, rtol=rtol, atol=atol)


# -------------------- compute_shannon_entropy --------------------

def test_shannon_entropy_uniform_two_bits_bits_base2():
    # Uniform over 2 bits → 4 states, each 1/4 → H = 2 bits
    probs = {"00": 0.25, "01": 0.25, "10": 0.25, "11": 0.25}
    H = compute_shannon_entropy(probs, total_prob=None, base=2)
    assert _isclose(H, 2.0)

def test_shannon_entropy_nats_vs_bits_conversion():
    # Single fair bit → H = 1 bit = ln(2) nats
    probs = {"0": 0.5, "1": 0.5}
    H_bits = compute_shannon_entropy(probs, total_prob=None, base=2)
    H_nats = compute_shannon_entropy(probs, total_prob=None, base=np.e)
    assert _isclose(H_bits, 1.0)
    assert _isclose(H_nats, math.log(2.0))

def test_shannon_entropy_accepts_unnormalized_when_total_none():
    # Values sum to 2; with total_prob=None we infer sum=2 and normalize internally
    probs = {"0": 1.0, "1": 1.0}
    H = compute_shannon_entropy(probs, total_prob=None, base=2)
    assert _isclose(H, 1.0)

def test_shannon_entropy_uses_explicit_total_prob_if_given():
    # Same dict but pass total_prob=2 explicitly; should match previous test
    probs = {"0": 1.0, "1": 1.0}
    H = compute_shannon_entropy(probs, total_prob=2.0, base=2)
    assert _isclose(H, 1.0)

def test_shannon_entropy_zero_or_negative_total_gives_zero():
    probs = {"0": 0.0, "1": 0.0}
    assert compute_shannon_entropy(probs, total_prob=None, base=2) == 0.0
    assert compute_shannon_entropy(probs, total_prob=0.0, base=2) == 0.0


# -------------------- compute_reduced_shannon_entropy --------------------

def test_reduced_entropy_matches_full_when_subsystem_is_all_indices():
    probs = {"00": 0.25, "01": 0.25, "10": 0.25, "11": 0.25}
    H_full = compute_shannon_entropy(probs, total_prob=None, base=2)
    H_sub = compute_reduced_shannon_entropy(probs, indices=[0, 1], base=2)
    assert _isclose(H_sub, H_full)

def test_reduced_entropy_on_single_bit_of_uniform_two_bits_is_one_bit():
    # Two bits uniform → each marginal bit is fair → H = 1 bit
    probs = {"00": 0.25, "01": 0.25, "10": 0.25, "11": 0.25}
    # pick MSB (index 0)
    H_A = compute_reduced_shannon_entropy(probs, indices=[0], base=2)
    assert _isclose(H_A, 1.0)

def test_reduced_entropy_configuration_vs_indices_equivalence():
    # Perfect correlation: states 00 and 11 with 0.5 each
    probs = {"00": 0.5, "11": 0.5}
    config = [0, 1]  # region A = index 0, region B = index 1
    H_A_cfg = compute_reduced_shannon_entropy(probs, configuration=config, target_region=0, base=2)
    H_A_idx = compute_reduced_shannon_entropy(probs, indices=[0], base=2)
    assert _isclose(H_A_cfg, H_A_idx)


# -------------------- compute_mutual_information --------------------

def test_mutual_information_zero_for_independent_bits():
    # Product distribution: P(A=0)=0.7, P(B=0)=0.4
    # Construct joint on two bits (MSB=A, LSB=B)
    pA0, pA1 = 0.7, 0.3
    pB0, pB1 = 0.4, 0.6
    probs = {
        "00": pA0 * pB0,
        "01": pA0 * pB1,
        "10": pA1 * pB0,
        "11": pA1 * pB1,
    }
    I, H_A, H_B, H_AB = compute_mutual_information(
        probs, configuration=[0, 1], base=2
    )
    # Independence ⇒ I(A:B)=0 and H(AB)=H(A)+H(B)
    assert _isclose(I, 0.0, atol=1e-12)
    assert _isclose(H_AB, H_A + H_B)

def test_mutual_information_positive_for_perfect_correlation():
    # Perfect correlation: states 00 and 11 with 0.5 each
    probs = {"00": 0.5, "11": 0.5}
    I, H_A, H_B, H_AB = compute_mutual_information(
        probs, configuration=[0, 1], base=2
    )
    # Each marginal is fair (H=1), joint has H=1 (two equiprobable states), so I = 1 + 1 - 1 = 1
    assert _isclose(H_A, 1.0)
    assert _isclose(H_B, 1.0)
    assert _isclose(H_AB, 1.0)
    assert _isclose(I, 1.0)

def test_mutual_information_indices_argument_path():
    # Use explicit indices instead of configuration
    probs = {"00": 0.5, "11": 0.5}
    I, H_A, H_B, H_AB = compute_mutual_information(
        probs, a_indices=[0], b_indices=[1], base=2
    )
    assert _isclose(I, 1.0)


# -------------------- compute_conditional_entropy --------------------

def test_conditional_entropy_relation_matches_definition():
    # Use a correlated example (same as above)
    probs = {"00": 0.5, "11": 0.5}
    # H(A|B) = H(AB) - H(B) = 1 - 1 = 0
    H_cond = compute_conditional_entropy(probs, configuration=[0, 1], base=2)
    assert _isclose(H_cond, 0.0)

def test_conditional_entropy_indices_path_only_B_needed():
    # Independent bits where H(AB)=H(A)+H(B)
    probs = {
        "00": 0.28, "01": 0.42, "10": 0.12, "11": 0.18  # pA0=0.7, pB0=0.4
    }
    # Provide B indices explicitly (index 1)
    H_AB = compute_shannon_entropy(probs, total_prob=None, base=2)
    # B marginal
    # Reduce to B: sum over MSB
    H_B = compute_reduced_shannon_entropy(probs, indices=[1], base=2)
    H_cond = compute_conditional_entropy(probs, b_indices=[1], base=2)
    assert _isclose(H_cond, H_AB - H_B)


# -------------------- Error handling --------------------

def test_errors_invalid_prob_dict_types_and_values():
    with pytest.raises(TypeError):
        compute_shannon_entropy([("0", 1.0)], total_prob=None)  # not a dict
    with pytest.raises(ValueError):
        compute_shannon_entropy({}, total_prob=None)  # empty dict
    with pytest.raises(ValueError):
        compute_shannon_entropy({"0X": 1.0}, total_prob=None)  # invalid key
    with pytest.raises(ValueError):
        compute_shannon_entropy({"0": -0.1, "1": 1.1}, total_prob=None)  # negative prob

def test_reduced_entropy_requires_configuration_or_indices():
    probs = {"00": 0.5, "11": 0.5}
    with pytest.raises(ValueError, match="configuration must be provided"):
        # Neither configuration nor indices
        compute_reduced_shannon_entropy(probs, configuration=None, indices=None)

def test_mutual_information_requires_config_or_both_indices():
    probs = {"00": 0.5, "11": 0.5}
    with pytest.raises(ValueError, match="Provide either"):
        compute_mutual_information(probs, configuration=None, a_indices=[0], b_indices=None)

def test_conditional_entropy_requires_config_or_b_indices():
    probs = {"00": 0.5, "11": 0.5}
    with pytest.raises(ValueError, match="Provide either"):
        compute_conditional_entropy(probs, configuration=None, b_indices=None)