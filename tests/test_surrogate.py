"""Tests for surrogate module (#82)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from trade_study import (
    Direction,
    Factor,
    FactorType,
    Observable,
    SurrogateModel,
    build_grid,
    fit_surrogate,
    run_grid,
)

if TYPE_CHECKING:
    from trade_study.protocols import ResultsTable


class _LinearWorld:
    """Trivial simulator: passes config through."""

    def generate(
        self, config: dict[str, float]
    ) -> tuple[dict[str, float], dict[str, float]]:
        return config, config


class _LinearScorer:
    """Scorer where ``y = 2*alpha + 0.5*beta`` and ``z = alpha**2``."""

    def score(
        self,
        truth: object,
        observations: dict[str, float],
        config: dict[str, float],
    ) -> dict[str, float]:
        del truth, config
        a = float(observations["alpha"])
        b = float(observations["beta"])
        return {"y": 2.0 * a + 0.5 * b, "z": a * a}


@pytest.fixture
def continuous_factors() -> list[Factor]:
    return [
        Factor("alpha", FactorType.CONTINUOUS, bounds=(0.0, 1.0)),
        Factor("beta", FactorType.CONTINUOUS, bounds=(0.0, 1.0)),
    ]


@pytest.fixture
def mixed_factors() -> list[Factor]:
    return [
        Factor("alpha", FactorType.CONTINUOUS, bounds=(0.0, 1.0)),
        Factor("kind", FactorType.CATEGORICAL, levels=["red", "green", "blue"]),
    ]


def _make_results(factors: list[Factor], n: int = 32, seed: int = 0) -> ResultsTable:
    grid = build_grid(factors, method="sobol", n_samples=n, seed=seed)
    obs = [
        Observable("y", Direction.MINIMIZE),
        Observable("z", Direction.MINIMIZE),
    ]
    return run_grid(_LinearWorld(), _LinearScorer(), grid, obs)


# ---------------------------------------------------------------------------
# Method dispatch and validation
# ---------------------------------------------------------------------------


def test_fit_surrogate_unknown_method(continuous_factors: list[Factor]) -> None:
    results = _make_results(continuous_factors, n=8)
    with pytest.raises(ValueError, match="Unknown surrogate method"):
        fit_surrogate(results, continuous_factors, method="bogus")


def test_fit_surrogate_empty_results(continuous_factors: list[Factor]) -> None:
    results = _make_results(continuous_factors, n=2)
    results.configs = []
    results.scores = np.empty((0, 2))
    with pytest.raises(ValueError, match="empty"):
        fit_surrogate(results, continuous_factors)


def test_fit_surrogate_all_nan(continuous_factors: list[Factor]) -> None:
    results = _make_results(continuous_factors, n=8)
    results.scores[:] = np.nan
    with pytest.raises(ValueError, match="non-NaN"):
        fit_surrogate(results, continuous_factors, method="rf")


# ---------------------------------------------------------------------------
# GP backend
# ---------------------------------------------------------------------------


def test_gp_predict_recovers_linear(continuous_factors: list[Factor]) -> None:
    results = _make_results(continuous_factors, n=64, seed=0)
    model = fit_surrogate(results, continuous_factors, method="gp", seed=0)
    pred = model.predict({"alpha": 0.5, "beta": 0.5})
    # Truth: y = 2*0.5 + 0.5*0.5 = 1.25, z = 0.25
    assert pred["y"] == pytest.approx(1.25, abs=0.05)
    assert pred["z"] == pytest.approx(0.25, abs=0.05)


def test_gp_uncertainty_returns_floats(continuous_factors: list[Factor]) -> None:
    results = _make_results(continuous_factors, n=32)
    model = fit_surrogate(results, continuous_factors, method="gp", seed=0)
    unc = model.uncertainty({"alpha": 0.5, "beta": 0.5})
    assert set(unc) == {"y", "z"}
    assert all(isinstance(v, float) and v >= 0.0 for v in unc.values())


def test_gp_predict_batch_shape(continuous_factors: list[Factor]) -> None:
    results = _make_results(continuous_factors, n=16)
    model = fit_surrogate(results, continuous_factors, method="gp", seed=0)
    batch = [{"alpha": a, "beta": 0.5} for a in [0.0, 0.25, 0.5, 0.75, 1.0]]
    pred = model.predict_batch(batch)
    assert pred["y"].shape == (5,)
    assert pred["z"].shape == (5,)


# ---------------------------------------------------------------------------
# RF backend
# ---------------------------------------------------------------------------


def test_rf_predict_runs(continuous_factors: list[Factor]) -> None:
    results = _make_results(continuous_factors, n=32)
    model = fit_surrogate(
        results,
        continuous_factors,
        method="rf",
        seed=0,
        n_estimators=50,
    )
    assert isinstance(model, SurrogateModel)
    pred = model.predict({"alpha": 0.5, "beta": 0.5})
    assert set(pred) == {"y", "z"}


def test_rf_uncertainty_raises(continuous_factors: list[Factor]) -> None:
    results = _make_results(continuous_factors, n=16)
    model = fit_surrogate(results, continuous_factors, method="rf", seed=0)
    with pytest.raises(NotImplementedError, match="method='gp'"):
        model.uncertainty({"alpha": 0.5, "beta": 0.5})


# ---------------------------------------------------------------------------
# Mixed (categorical) factor encoding
# ---------------------------------------------------------------------------


def test_mixed_factor_encoding(mixed_factors: list[Factor]) -> None:
    grid = build_grid(mixed_factors, method="sobol", n_samples=24, seed=0)
    obs = [Observable("y", Direction.MINIMIZE)]

    class _MixedScorer:
        def score(
            self,
            truth: object,
            observations: dict[str, object],
            config: dict[str, object],
        ) -> dict[str, float]:
            del truth, config
            offset = {"red": 0.0, "green": 1.0, "blue": -1.0}[str(observations["kind"])]
            return {"y": float(observations["alpha"]) + offset}

    results = run_grid(_LinearWorld(), _MixedScorer(), grid, obs)
    model = fit_surrogate(results, mixed_factors, method="rf", seed=0)
    pred_red = model.predict({"alpha": 0.5, "kind": "red"})["y"]
    pred_green = model.predict({"alpha": 0.5, "kind": "green"})["y"]
    pred_blue = model.predict({"alpha": 0.5, "kind": "blue"})["y"]
    # Order should be preserved by the encoding even with a coarse RF.
    assert pred_blue < pred_red < pred_green


class _AlphaScorer:
    """Scorer that only depends on ``alpha``."""

    def score(
        self,
        truth: object,
        observations: dict[str, object],
        config: dict[str, object],
    ) -> dict[str, float]:
        del truth, config
        return {"y": float(observations["alpha"])}


def test_predict_rejects_unknown_level(mixed_factors: list[Factor]) -> None:
    grid = build_grid(mixed_factors, method="sobol", n_samples=8, seed=0)
    obs = [Observable("y", Direction.MINIMIZE)]
    results = run_grid(_LinearWorld(), _AlphaScorer(), grid, obs)
    model = fit_surrogate(results, mixed_factors, method="rf", seed=0)
    with pytest.raises(ValueError, match="not in declared levels"):
        model.predict({"alpha": 0.5, "kind": "purple"})


def test_predict_rejects_missing_factor(mixed_factors: list[Factor]) -> None:
    grid = build_grid(mixed_factors, method="sobol", n_samples=8, seed=0)
    obs = [Observable("y", Direction.MINIMIZE)]
    results = run_grid(_LinearWorld(), _AlphaScorer(), grid, obs)
    model = fit_surrogate(results, mixed_factors, method="rf", seed=0)
    with pytest.raises(KeyError, match="missing factor"):
        model.predict({"alpha": 0.5})


# ---------------------------------------------------------------------------
# NaN handling
# ---------------------------------------------------------------------------


def test_partial_nan_drops_rows(continuous_factors: list[Factor]) -> None:
    results = _make_results(continuous_factors, n=32)
    # Knock out half the rows for observable "y" only.
    results.scores[::2, 0] = np.nan
    model = fit_surrogate(results, continuous_factors, method="rf", seed=0)
    pred = model.predict({"alpha": 0.5, "beta": 0.5})
    # Both observables should still be predictable since z has no NaNs.
    assert set(pred) == {"y", "z"}
