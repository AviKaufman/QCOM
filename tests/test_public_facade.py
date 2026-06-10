import qcom


def test_top_level_lazy_exports_are_stable():
    assert qcom.__version__ == "0.2.2"
    assert qcom.LatticeRegister is not None
    assert qcom.TimeSeries is not None
    assert qcom.CountsData is not None
    assert qcom.MutualInformationResult is not None
    assert callable(qcom.build_ising)
    assert callable(qcom.build_rydberg)
    assert callable(qcom.compute_mutual_information)


def test_submodules_are_lazy_accessible():
    assert qcom.core.CountsData is qcom.CountsData
    assert qcom.hamiltonians.build_ising is not None
    assert qcom.viz.plot_lattice_register is not None
