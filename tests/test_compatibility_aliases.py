from __future__ import annotations

import random
import warnings
from pathlib import Path

import numpy as np

from qcom.data.noise import apply_readout_error, introduce_error
from qcom.data.sampling import (
    combine_bitstring_datasets,
    combine_datasets,
    sample_counts,
    sample_data,
)
from qcom.io.aquila import parse_aquila_json, parse_json
from qcom.io.text import parse_file, parse_text, save_data, save_text
from qcom.metrics.bitstrings import (
    marginalize_bitstring_distribution,
    order_dict,
    part_dict,
    sort_bitstring_distribution,
)
from qcom.metrics.probabilities import (
    compute_cumulative_distribution,
    compute_cumulative_probability_at_value,
    cumulative_distribution,
    cumulative_probability_at_value,
)


def _assert_deprecated_once(
    result_fn,
    alias_fn,
    *args,
    preferred_kwargs: dict[str, object] | None = None,
    **kwargs,
):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        alias_result = alias_fn(*args, **kwargs)
    assert len(caught) == 1
    assert issubclass(caught[0].category, DeprecationWarning)
    return alias_result, result_fn(*args, **(preferred_kwargs or kwargs))


def test_compatibility_aliases_warn_once_and_forward_kwargs(tmp_path: Path) -> None:
    bitstring_values = {"10": 0.1, "01": 0.2, "00": 0.3, "11": 0.4}
    probabilities = {"0": 0.5, "1": 0.5}

    alias_result, preferred_result = _assert_deprecated_once(
        sort_bitstring_distribution,
        order_dict,
        bitstring_values,
    )
    assert alias_result == preferred_result

    alias_result, preferred_result = _assert_deprecated_once(
        marginalize_bitstring_distribution,
        part_dict,
        bitstring_values,
        [0],
    )
    assert alias_result == preferred_result

    alias_result, preferred_result = _assert_deprecated_once(
        compute_cumulative_probability_at_value,
        cumulative_probability_at_value,
        probabilities,
        0.5,
    )
    assert np.isclose(alias_result, preferred_result)

    alias_result, preferred_result = _assert_deprecated_once(
        compute_cumulative_distribution,
        cumulative_distribution,
        probabilities,
        grid=[0.5, 1.0],
    )
    assert np.allclose(alias_result[0], preferred_result[0])
    assert np.allclose(alias_result[1], preferred_result[1])

    random.seed(12345)
    alias_result, preferred_result = _assert_deprecated_once(
        sample_counts,
        sample_data,
        {"0": 1},
        total_count=1,
        sample_size=3,
    )
    assert alias_result == preferred_result

    alias_result, preferred_result = _assert_deprecated_once(
        combine_bitstring_datasets,
        combine_datasets,
        {"0": 2},
        {"0": 3},
    )
    assert alias_result == preferred_result

    alias_result, preferred_result = _assert_deprecated_once(
        apply_readout_error,
        introduce_error,
        {"00": 2},
        ground_rate=0.0,
        excited_rate=0.0,
        seed=1,
    )
    assert alias_result == preferred_result

    text_path = tmp_path / "counts.txt"
    alias_path = tmp_path / "alias.txt"
    preferred_path = tmp_path / "preferred.txt"
    save_text({"00": 1.0, "11": 2.0}, str(preferred_path), show_progress=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        save_data({"00": 1.0, "11": 2.0}, str(alias_path), show_progress=False)
    assert len(caught) == 1
    assert alias_path.read_text() == preferred_path.read_text()

    text_path.write_text("00 1.0\n11 2.0\n", encoding="utf-8")
    alias_result, preferred_result = _assert_deprecated_once(
        parse_text,
        parse_file,
        str(text_path),
        show_progress=False,
    )
    assert alias_result == preferred_result

    aquila_path = tmp_path / "aquila.json"
    aquila_path.write_text(
        """{\"measurements\": [{\"shotResult\": {\"preSequence\": [1, 1], \"postSequence\": [0, 1]}}]}""",
        encoding="utf-8",
    )
    alias_result, preferred_result = _assert_deprecated_once(
        parse_aquila_json,
        parse_json,
        str(aquila_path),
        sorted=False,
        filter_incomplete=False,
        show_progress=False,
        preferred_kwargs={"filter_incomplete": False, "show_progress": False},
    )
    assert alias_result == preferred_result
