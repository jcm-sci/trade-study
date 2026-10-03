"""DataFrame exports preserve results and statistical metadata (#85)."""

from __future__ import annotations

import builtins
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from trade_study import ResultsTable

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


def _table() -> ResultsTable:
    return ResultsTable(
        configs=[{"method": "a", "x": 1}, {"method": "b", "x": 2}],
        scores=np.array([[0.1], [np.nan]]),
        observable_names=["loss"],
        annotations=np.array([[10.0], [20.0]]),
        annotation_names=["cost"],
        metadata=[
            {
                "rep": 0,
                "design_point": 0,
                "standard_error": {"loss": 0.01},
                "n_reps": {"loss": 3},
                "scores": {"loss": 0.05},
            },
            {"rep": 1, "design_point": 0},
        ],
    )


def test_export_preserves_order_values_and_nested_metadata() -> None:
    table = _table()
    frame = table.to_dataframe()
    assert list(frame.columns[:4]) == ["method", "x", "loss", "cost"]
    assert frame["method"].tolist() == ["a", "b"]
    assert frame["meta.rep"].tolist() == [0, 1]
    assert frame.loc[0, "meta.standard_error.loss"] == pytest.approx(0.01)
    assert frame.loc[0, "meta.n_reps.loss"] == 3
    assert frame.loc[0, "meta.scores.loss"] == pytest.approx(0.05)
    assert np.isnan(frame.loc[1, "loss"])
    frame.loc[0, "loss"] = 9.0
    assert table.scores[0, 0] == pytest.approx(0.1)
    frame.loc[0, "cost"] = 99.0
    assert table.annotations is not None
    assert table.annotations[0, 0] == pytest.approx(10.0)


def test_metadata_can_be_omitted() -> None:
    assert list(_table().to_dataframe(include_metadata=False).columns) == [
        "method",
        "x",
        "loss",
        "cost",
    ]


def test_sparse_configs_and_empty_tables() -> None:
    table = ResultsTable(
        configs=[{"a": "one"}, {"b": 2}],
        scores=np.array([[1.0], [2.0]]),
        observable_names=["loss"],
    )
    frame = table.to_dataframe()
    assert list(frame.columns) == ["a", "b", "loss"]
    assert frame.loc[0, "a"] == "one"
    assert np.isnan(frame.loc[0, "b"])
    empty = ResultsTable(configs=[], scores=np.empty((0, 1)), observable_names=["loss"])
    assert list(empty.to_dataframe().columns) == ["loss"]
    assert empty.to_dataframe().empty


@pytest.mark.parametrize("name", ["loss", "cost", "meta.rep"])
def test_column_collisions_raise(name: str) -> None:
    table = _table()
    table.configs[0][name] = 1
    with pytest.raises(ValueError, match="collide"):
        table.to_dataframe()


def test_metadata_length_is_validated() -> None:
    table = _table()
    table.metadata = [{}]
    with pytest.raises(ValueError, match="one entry per trial"):
        table.to_dataframe()


def test_missing_pandas_has_install_guidance(monkeypatch: pytest.MonkeyPatch) -> None:
    original = builtins.__import__

    def unavailable(
        name: str,
        globals_dict: Mapping[str, Any] | None = None,
        locals_dict: Mapping[str, Any] | None = None,
        fromlist: Sequence[str] = (),
        level: int = 0,
    ) -> Any:
        if name == "pandas":
            msg = "pandas intentionally unavailable"
            raise ImportError(msg)
        return original(name, globals_dict, locals_dict, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", unavailable)
    with pytest.raises(ImportError, match=r"trade-study\[dataframe\]"):
        _table().to_dataframe()
