import qcom


_TOP_LEVEL_SUBMODULES = {
    "controls",
    "core",
    "data",
    "hamiltonians",
    "io",
    "metrics",
    "solvers",
    "viz",
}

_TOP_LEVEL_COMPATIBILITY_ALIASES = {
    "combine_datasets",
    "cumulative_distribution",
    "introduce_error",
    "parse_file",
    "parse_json",
    "print_most_probable_data",
    "sample_data",
    "save_data",
}

_PREFERRED_PUBLIC_NAMES = {
    "apply_readout_error",
    "build_ising",
    "build_rydberg",
    "combine_bitstring_datasets",
    "compute_conditional_entropy",
    "compute_cumulative_distribution",
    "compute_cumulative_probability_at_value",
    "compute_mutual_information",
    "compute_n_of_p",
    "compute_n_of_p_curve",
    "compute_reduced_shannon_entropy",
    "compute_shannon_entropy",
    "find_eigenstate",
    "get_eigenstate_probabilities",
    "ground_state",
    "m3_mitigate_counts_from_rates",
    "marginalize_bitstring_distribution",
    "normalize_to_probabilities",
    "parse_aquila_json",
    "parse_parquet",
    "parse_text",
    "print_most_probable_bitstrings",
    "sample_counts",
    "save_parquet",
    "save_text",
    "sort_bitstring_distribution",
    "statevector_to_probabilities",
    "truncate_probabilities",
    "von_neumann_entropy_from_state",
}


def test_top_level_lazy_exports_are_stable():
    assert qcom.__version__ == "0.2.2"
    assert qcom.LatticeRegister is not None
    assert qcom.TimeSeries is not None
    assert qcom.CountsData is not None
    assert qcom.MutualInformationResult is not None
    assert callable(qcom.apply_readout_error)
    assert callable(qcom.build_ising)
    assert callable(qcom.build_rydberg)
    assert callable(qcom.combine_bitstring_datasets)
    assert callable(qcom.compute_cumulative_distribution)
    assert callable(qcom.compute_cumulative_probability_at_value)
    assert callable(qcom.compute_mutual_information)
    assert callable(qcom.compute_n_of_p)
    assert callable(qcom.compute_n_of_p_curve)
    assert callable(qcom.marginalize_bitstring_distribution)
    assert callable(qcom.parse_aquila_json)
    assert callable(qcom.parse_text)
    assert callable(qcom.print_most_probable_bitstrings)
    assert callable(qcom.sample_counts)
    assert callable(qcom.save_parquet)
    assert callable(qcom.save_text)
    assert callable(qcom.sort_bitstring_distribution)


def test_submodules_are_lazy_accessible():
    assert qcom.core.CountsData is qcom.CountsData
    assert qcom.hamiltonians.build_ising is not None
    assert qcom.viz.plot_lattice_register is not None


def test_top_level_public_api_exports_are_intentional():
    public_names = set(qcom.__all__)

    assert _PREFERRED_PUBLIC_NAMES <= public_names
    assert _TOP_LEVEL_COMPATIBILITY_ALIASES <= public_names

    for name in sorted(public_names - {"__version__"}):
        value = getattr(qcom, name)
        if name not in _TOP_LEVEL_SUBMODULES:
            assert value is not None
