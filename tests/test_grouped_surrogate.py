"""Validation leakage and observed-support diagnostics for both backends."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from trade_study import (
    Factor,
    FactorType,
    ResultsTable,
    fit_regime_surrogate,
    fit_surrogate,
)


@pytest.mark.parametrize("method", ["gp", "rf"])
def test_held_out_designs_do_not_share_replicates(method: str) -> None:
    factors = [Factor("design", FactorType.CATEGORICAL, levels=list(range(6)))]
    results = ResultsTable(
        configs=[{"design": i} for i in range(6) for _ in range(12)],
        scores=np.repeat([-3, -2, -1, 1, 2, 3], 12).reshape(-1, 1),
        observable_names=["loss"],
    )
    fitted = fit_surrogate(
        results,
        factors,
        method=method,
        n_estimators=30,
        cv_folds=3,
        cv_group_by="design",
        warn_below_r2=None,
    )
    assert fitted.cv_group_by == ("design",)
    assert fitted.validation_groups == [i for i in range(6) for _ in range(12)]
    assert fitted.row_cv_r2["loss"] > 0.95
    assert fitted.cv_r2["loss"] < fitted.row_cv_r2["loss"] - 0.5


@pytest.mark.parametrize("method", ["gp", "rf"])
def test_regime_groups_and_recommendation_support(method: str) -> None:
    regime_factors = [Factor("noise", FactorType.CONTINUOUS, bounds=(0.0, 2.0))]
    factors = [Factor("x", FactorType.CONTINUOUS, bounds=(0.0, 1.0))]
    results = ResultsTable(
        configs=[{"noise": n, "x": x} for n in (0.5, 1.0) for x in (0.2, 0.5, 0.8)],
        scores=np.array([[n + x] for n in (0.5, 1.0) for x in (0.2, 0.5, 0.8)]),
        observable_names=["loss"],
    )
    model = fit_regime_surrogate(
        results,
        regime_factors,
        factors,
        method=method,
        n_estimators=20,
        cv_group_by="regime",
        warn_below_r2=None,
    )
    assert model.inner.cv_group_by == ("noise",)
    assert model.inner.validation_groups == [0, 0, 0, 1, 1, 1]
    assert model.row_cv_r2 == model.inner.row_cv_r2
    assert model.row_cv_rmse == model.inner.row_cv_rmse
    candidates = [{"x": 0.5}]
    assert model.support({"noise": 0.75}, candidates)[0].supported
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        assert model.recommend(
            {"noise": 0.75},
            objective="loss",
            candidates=candidates,
            warn_below_r2=None,
        ) == {"x": 0.5}
    outside = model.support({"noise": 1.5}, candidates)[0]
    assert outside.outside_ranges == {"noise": (0.5, 1.0)}
    with pytest.warns(UserWarning, match="outside observed training support"):
        choice = model.recommend(
            {"noise": 1.5},
            objective="loss",
            candidates=candidates,
            warn_below_r2=None,
            risk=1,
        )
    assert choice == {"x": 0.5}
    assert model.predict({"noise": 1.5}, choice, warn_support=False)[
        "loss"
    ] == pytest.approx(
        model.inner.predict({"noise": 1.5, "x": 0.5}, warn_support=False)["loss"]
    )


@pytest.mark.parametrize("method", ["gp", "rf"])
def test_support_uses_finite_rows_per_observable_and_raw_log_units(method: str) -> None:
    factors = [
        Factor("x", FactorType.CONTINUOUS, bounds=(0.01, 100), log_scale=True),
        Factor("mode", FactorType.CATEGORICAL, levels=["a", "b", "c"]),
    ]
    results = ResultsTable(
        configs=[
            {"x": 0.1, "mode": "a"},
            {"x": 1, "mode": "a"},
            {"x": 10, "mode": "b"},
        ],
        scores=np.array([[0, 0], [1, 1], [np.nan, 2]]),
        observable_names=["sparse", "full"],
    )
    model = fit_surrogate(
        results, factors, method=method, n_estimators=20, warn_below_r2=None
    )
    reports = model.support([{"x": 10, "mode": "b"}])
    assert reports[0].outside_ranges == {"x": (0.1, 1.0)}
    assert reports[0].unseen_levels == {"mode": "b"}
    assert reports[1].supported
    config = {"x": 0.01, "mode": "c"}
    with pytest.warns(UserWarning, match="training support"):
        model.predict(config)
    with pytest.warns(UserWarning, match="training support"):
        model.spread_batch([config])
    if method == "gp":
        with pytest.warns(UserWarning, match="training support"):
            model.uncertainty(config)
    assert all(
        np.isfinite(v) for v in model.predict(config, warn_support=False).values()
    )
    with pytest.raises(ValueError, match="declared levels"):
        model.predict({"x": 1, "mode": "undeclared"})


def test_sparse_groups_and_constant_targets_report_nan_r2() -> None:
    factors = [Factor("x", FactorType.CONTINUOUS, bounds=(0, 1))]
    results = ResultsTable(
        configs=[{"x": 0.5}] * 3,
        scores=np.array([[1, np.inf], [1, 2], [1, np.nan]]),
        observable_names=["constant", "sparse"],
    )
    model = fit_surrogate(results, factors, method="rf", cv_group_by="design")
    assert model.observable_names == ["constant"]
    assert np.isnan(model.cv_r2["constant"])
    assert np.isnan(model.cv_rmse["constant"])
    assert np.isnan(model.row_cv_r2["constant"])
    assert model.row_cv_rmse["constant"] == 0


def test_default_retains_row_cv() -> None:
    factors = [Factor("x", FactorType.CONTINUOUS, bounds=(0, 1))]
    result = ResultsTable(
        [{"x": x} for x in (0, 0.5, 1)], np.array([[0], [1], [2]]), ["y"]
    )
    model = fit_surrogate(result, factors, method="rf", warn_below_r2=None)
    assert model.cv_group_by is None
    assert model.validation_groups == []
    assert model.cv_r2 == model.row_cv_r2
    assert model.cv_rmse == model.row_cv_rmse


def test_numerically_identical_designs_share_a_group() -> None:
    results = ResultsTable(
        [{"x": 0, "category": 1}, {"x": 0.0, "category": 1.0}, {"x": 1, "category": 2}],
        np.array([[0.0], [0.0], [1.0]]),
        ["y"],
    )
    model = fit_surrogate(
        results,
        [
            Factor("x", FactorType.CONTINUOUS, bounds=(0, 1)),
            Factor("category", FactorType.CATEGORICAL, levels=[1, 2]),
        ],
        method="rf",
        cv_group_by="design",
        warn_below_r2=None,
    )
    assert model.validation_groups == [0, 0, 1]


@pytest.mark.parametrize("grouping", [[], ["missing"], ["x", "x"], "unknown"])
def test_invalid_grouping_is_rejected(grouping: list[str] | str) -> None:
    result = ResultsTable([{"x": 0}, {"x": 1}], np.array([[0], [1]]), ["y"])
    with pytest.raises(ValueError, match="cv_group_by"):
        fit_surrogate(
            result,
            [Factor("x", FactorType.CONTINUOUS, bounds=(0, 1))],
            cv_group_by=grouping,
        )
