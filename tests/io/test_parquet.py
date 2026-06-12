import pytest

pytest.importorskip("pyarrow")

import pandas as pd
from pathlib import Path

from qcom.io.parquet import (
    _install_pyarrow_hotfix,
    parse_parquet,
    save_dict_to_parquet,
    save_parquet,
)


def test_roundtrip_save_and_parse(tmp_path: Path):
    data = {"00": 0.5, "01": 0.25, "10": 0.25}
    fn = tmp_path / "dist.parquet"

    save_parquet(data, str(fn))

    parsed = parse_parquet(str(fn), show_progress=False)

    assert parsed == data
    assert all(isinstance(v, float) for v in parsed.values())


def test_parse_parquet_written_by_pandas(tmp_path: Path):
    df = pd.DataFrame({"state": ["000", "111", "101"], "probability": [0.2, 0.5, 0.3]})
    fn = tmp_path / "pandas_written.parquet"
    df.to_parquet(fn, engine="pyarrow", index=False)

    parsed = parse_parquet(str(fn), show_progress=False)
    expected = {"000": 0.2, "111": 0.5, "101": 0.3}

    assert parsed == expected


def test_parse_parquet_missing_columns_raises_keyerror(tmp_path: Path):
    df = pd.DataFrame({"bit": ["0", "1"], "p": [0.4, 0.6]})
    fn = tmp_path / "bad_schema.parquet"
    df.to_parquet(fn, engine="pyarrow", index=False)

    with pytest.raises(KeyError):
        _ = parse_parquet(str(fn), show_progress=False)


def test_save_dict_to_parquet_compatibility_alias(tmp_path: Path):
    data = {"0": 1.0}
    fn = tmp_path / "compat.parquet"
    with pytest.warns(DeprecationWarning, match="save_parquet"):
        save_dict_to_parquet(data, str(fn))
    assert parse_parquet(str(fn), show_progress=False) == data


def test_parse_parquet_file_name_keyword_warns(tmp_path: Path):
    data = {"0": 1.0}
    fn = tmp_path / "file_name.parquet"
    save_parquet(data, str(fn))

    with pytest.warns(DeprecationWarning, match="file_path"):
        assert parse_parquet(file_name=str(fn), show_progress=False) == data


def test_parse_parquet_handles_missing_pyarrow_extension_type(monkeypatch):
    import pyarrow

    calls: list[str] = []

    def fake_unregister_extension_type(name: str):
        calls.append(name)
        if name == "arrow.py_extension_type":
            raise pyarrow.ArrowKeyError(name)
        return None

    monkeypatch.setattr(pyarrow, "unregister_extension_type", fake_unregister_extension_type)
    monkeypatch.delattr(pyarrow, "_qcom_hotfix_installed", raising=False)

    _install_pyarrow_hotfix()
    pyarrow.unregister_extension_type("arrow.py_extension_type")

    assert "arrow.py_extension_type" in calls
