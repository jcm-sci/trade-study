"""Tests for post-hoc sensitivity from an existing ResultsTable (#113)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from trade_study import (
    Direction,
    Factor,
    FactorType,
    Observable,
    TableSensitivity,
    build_grid,
    run_grid,
    sensitivity_from_table,
)

if TYPE_CHECKING:
    from trade_study.protocols import ResultsTable


class _World:
    """Trivial simulator: passes config through."""

    def generate(
        self, config: dict[str, float]
    ) -> tuple[dict[str, float], dict[str, float]]:
        return config, config


class _NonMonotonicScorer:
    """Scorer where ``y = (a - 0.5)**2 + 0.1*b``.

    ``a`` has a U-shaped (non-monotonic) effect symmetric about its
    midpoint -- a marginal Spearman correlation would show it as
    near-zero despite it being the dominant driver of variance. ``b`` has
    a small, purely linear effect.
    """

    def score(
        self,
        truth: object,
        observations: dict[str, float],
        config: dict[str, float],
    ) -> dict[str, float]:
        del truth, config
        a = float(observations["a"])
        b = float(observations["b"])
        return {"y": (a - 0.5) ** 2 + 0.1 * b}


@pytest.fixture
def continuous_factors() -> list[Factor]:
    return [
        Factor("a", FactorType.CONTINUOUS, bounds=(0.0, 1.0)),
        Factor("b", FactorType.CONTINUOUS, bounds=(0.0, 1.0)),
    ]


def _make_results(factors: list[Factor], n: int = 128, seed: int = 0) -> ResultsTable:
    grid = build_grid(factors, method="sobol", n_samples=n, seed=seed)
    obs = [Observable("y", Direction.MINIMIZE)]
    return run_grid(_World(), _NonMonotonicScorer(), grid, obs)


def test_returns_table_sensitivity(continuous_factors: list[Factor]) -> None:
    results = _make_results(continuous_factors)
    result = sensitivity_from_table(results, continuous_factors, seed=0)
    assert isinstance(result, TableSensitivity)
    assert "y" in result.importance
    assert "y" in result.surrogate_cv_r2


def test_importance_shape(continuous_factors: list[Factor]) -> None:
    results = _make_results(continuous_factors)
    result = sensitivity_from_table(results, continuous_factors, seed=0)
    assert result.importance["y"].shape == (2,)


def test_sobol_detects_nonmonotonic_effect(continuous_factors: list[Factor]) -> None:
    """Sobol S1 (via the surrogate) should rank U-shaped 'a' above linear 'b'.

    This is the exact failure mode #113 was filed over: a marginal
    Spearman correlation misses this because 'a's effect is symmetric
    around its midpoint.
    """
    results = _make_results(continuous_factors, n=128)
    result = sensitivity_from_table(
        results, continuous_factors, method="sobol", surrogate_method="rf", seed=0
    )
    importance = result.importance["y"]
    assert importance[0] > importance[1]
    # 'a' should be clearly dominant, not just marginally ahead.
    assert importance[0] > 0.5


def test_morris_method_runs(continuous_factors: list[Factor]) -> None:
    results = _make_results(continuous_factors)
    result = sensitivity_from_table(
        results, continuous_factors, method="morris", n_trajectories=20, seed=0
    )
    assert result.importance["y"].shape == (2,)


def test_surrogate_cv_r2_reflects_fit_quality(continuous_factors: list[Factor]) -> None:
    results = _make_results(continuous_factors, n=128)
    result = sensitivity_from_table(results, continuous_factors, seed=0)
    assert result.surrogate_cv_r2["y"] > 0.5


def test_drops_noncontinuous_factors(continuous_factors: list[Factor]) -> None:
    """A categorical factor in the input list is dropped, not an error."""
    mixed = [*continuous_factors, Factor("kind", FactorType.CATEGORICAL, levels=["x"])]
    results = _make_results(continuous_factors)
    result = sensitivity_from_table(results, mixed, seed=0)
    # Only the two continuous factors are screened.
    assert result.importance["y"].shape == (2,)


def test_propagates_screen_no_continuous_error(
    continuous_factors: list[Factor],
) -> None:
    factors = [Factor("kind", FactorType.CATEGORICAL, levels=["a", "b"])]
    results = _make_results(continuous_factors, n=8)
    with pytest.raises(ValueError, match="at least one continuous"):
        sensitivity_from_table(results, factors, seed=0)


def test_warns_on_poor_surrogate_fit(continuous_factors: list[Factor]) -> None:
    """A surrogate fit to pure noise warns via the forwarded fit_surrogate call."""
    results = _make_results(continuous_factors, n=16)
    rng = np.random.default_rng(0)
    results.scores[:, 0] = rng.standard_normal(len(results.configs))
    with pytest.warns(UserWarning, match="cross-validated R\\^2 below"):
        sensitivity_from_table(results, continuous_factors, seed=0)
