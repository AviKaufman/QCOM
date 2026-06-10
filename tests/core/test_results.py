import numpy as np
import pytest

from qcom.core import CountsData, EvolutionResult, ProbabilityData, SpectrumResult


def test_counts_data_infers_shots_and_sites_and_is_immutable():
    data = CountsData({"00": 2, "11": 3})

    assert data.shots == 5
    assert data.n_sites == 2
    assert data.to_dict() == {"00": 2, "11": 3}
    with pytest.raises(TypeError):
        data.counts["00"] = 99


def test_counts_data_rejects_inconsistent_metadata():
    with pytest.raises(ValueError, match="shots"):
        CountsData({"0": 1}, shots=2)
    with pytest.raises(ValueError, match="n_sites"):
        CountsData({"0": 1}, n_sites=2)


def test_probability_data_validates_normalization():
    data = ProbabilityData({"0": 0.25, "1": 0.75})

    assert data.n_sites == 1
    assert data.to_dict() == {"0": 0.25, "1": 0.75}
    with pytest.raises(ValueError, match="sum"):
        ProbabilityData({"0": 0.25, "1": 0.25})


def test_result_containers_coerce_arrays_and_metadata():
    evo = EvolutionResult(final_state=[1, 0], times=[0, 1], states=([1, 0], [0, 1]))
    spec = SpectrumResult(eigenvalues=[0.0, 1.0], eigenvectors=np.eye(2))

    assert evo.final_state.dtype == np.complex128
    assert evo.times.shape == (2,)
    assert len(evo.states) == 2
    assert spec.eigenvalues.shape == (2,)
    assert spec.eigenvectors.shape == (2, 2)
