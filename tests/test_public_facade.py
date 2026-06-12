import qcom


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
