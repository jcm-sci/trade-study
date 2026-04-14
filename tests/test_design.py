"""Tests for design module (issues #8, #9, #10)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from trade_study.design import Factor, FactorType, build_grid, reduce_factors, screen

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def categorical_factors() -> list[Factor]:
    """Two categorical factors for full-factorial tests.

    Returns:
        List of two categorical factors.
    """
    return [
        Factor("method", FactorType.CATEGORICAL, levels=["a", "b"]),
        Factor("variant", FactorType.CATEGORICAL, levels=["x", "y", "z"]),
    ]


@pytest.fixture
def continuous_factors() -> list[Factor]:
    """Two continuous factors for LHS / screening tests.

    Returns:
        List of two continuous factors.
    """
    return [
        Factor("alpha", FactorType.CONTINUOUS, bounds=(0.0, 1.0)),
        Factor("beta", FactorType.CONTINUOUS, bounds=(10.0, 20.0)),
    ]


@pytest.fixture
def mixed_factors() -> list[Factor]:
    """Continuous + categorical factors for mixed-grid tests.

    Returns:
        List with one continuous and one categorical factor.
    """
    return [
        Factor("alpha", FactorType.CONTINUOUS, bounds=(0.0, 1.0)),
        Factor("colour", FactorType.CATEGORICAL, levels=["red", "green", "blue"]),
    ]


# ---------------------------------------------------------------------------
# Factor validation
# ---------------------------------------------------------------------------


def test_continuous_factor_requires_bounds() -> None:
    with pytest.raises(ValueError, match="requires bounds"):
        Factor("x", FactorType.CONTINUOUS)


def test_categorical_factor_requires_levels() -> None:
    with pytest.raises(ValueError, match="requires levels"):
        Factor("x", FactorType.CATEGORICAL)


def test_discrete_factor_requires_levels() -> None:
    with pytest.raises(ValueError, match="requires levels"):
        Factor("x", FactorType.DISCRETE)


# ---------------------------------------------------------------------------
# build_grid — full factorial (#8)
# ---------------------------------------------------------------------------


def test_full_factorial_count(categorical_factors: list[Factor]) -> None:
    grid = build_grid(categorical_factors, method="full")
    assert len(grid) == 2 * 3


def test_full_factorial_keys(categorical_factors: list[Factor]) -> None:
    grid = build_grid(categorical_factors, method="full")
    assert all(set(cfg.keys()) == {"method", "variant"} for cfg in grid)


def test_full_factorial_all_combos(categorical_factors: list[Factor]) -> None:
    grid = build_grid(categorical_factors, method="full")
    combos = {(cfg["method"], cfg["variant"]) for cfg in grid}
    expected = {("a", "x"), ("a", "y"), ("a", "z"), ("b", "x"), ("b", "y"), ("b", "z")}
    assert combos == expected


def test_full_factorial_rejects_continuous_bounds(
    continuous_factors: list[Factor],
) -> None:
    with pytest.raises(ValueError, match="requires levels"):
        build_grid(continuous_factors, method="full")


def test_full_factorial_discrete() -> None:
    factors = [Factor("n", FactorType.DISCRETE, levels=[1, 2, 3])]
    grid = build_grid(factors, method="full")
    assert [cfg["n"] for cfg in grid] == [1, 2, 3]


# ---------------------------------------------------------------------------
# build_grid — LHS (#8)
# ---------------------------------------------------------------------------


def test_lhs_sample_count(continuous_factors: list[Factor]) -> None:
    grid = build_grid(continuous_factors, method="lhs", n_samples=50)
    assert len(grid) == 50


def test_lhs_continuous_bounds(continuous_factors: list[Factor]) -> None:
    grid = build_grid(continuous_factors, method="lhs", n_samples=200)
    alphas = [cfg["alpha"] for cfg in grid]
    betas = [cfg["beta"] for cfg in grid]
    assert all(0.0 <= a <= 1.0 for a in alphas)
    assert all(10.0 <= b <= 20.0 for b in betas)


def test_lhs_categorical_in_levels(mixed_factors: list[Factor]) -> None:
    grid = build_grid(mixed_factors, method="lhs", n_samples=100)
    colours = {cfg["colour"] for cfg in grid}
    assert colours <= {"red", "green", "blue"}


def test_lhs_deterministic_with_seed(continuous_factors: list[Factor]) -> None:
    g1 = build_grid(continuous_factors, method="lhs", n_samples=20, seed=99)
    g2 = build_grid(continuous_factors, method="lhs", n_samples=20, seed=99)
    assert g1 == g2


def test_lhs_different_seeds_differ(continuous_factors: list[Factor]) -> None:
    g1 = build_grid(continuous_factors, method="lhs", n_samples=20, seed=1)
    g2 = build_grid(continuous_factors, method="lhs", n_samples=20, seed=2)
    assert g1 != g2


def test_build_grid_unknown_method(continuous_factors: list[Factor]) -> None:
    with pytest.raises(ValueError, match="Unknown design method"):
        build_grid(continuous_factors, method="bogus")


# ---------------------------------------------------------------------------
# screen — Morris (#9)
# ---------------------------------------------------------------------------


def _linear_model(cfg: dict[str, Any]) -> dict[str, float]:
    """Toy model: y = 3*alpha + 0*beta (beta is inert).

    Returns:
        Single-observable dict with key ``"y"``.
    """
    return {"y": 3.0 * cfg["alpha"] + 0.0 * cfg["beta"]}


def test_screen_returns_dict(continuous_factors: list[Factor]) -> None:
    result = screen(_linear_model, continuous_factors, n_trajectories=20, seed=0)
    assert isinstance(result, dict)
    assert "y" in result


def test_screen_importance_shape(continuous_factors: list[Factor]) -> None:
    result = screen(_linear_model, continuous_factors, n_trajectories=20, seed=0)
    assert result["y"].shape == (2,)


def test_screen_detects_influential_factor(continuous_factors: list[Factor]) -> None:
    result = screen(_linear_model, continuous_factors, n_trajectories=50, seed=0)
    # alpha (index 0) should dominate; beta (index 1) should be near zero
    assert result["y"][0] > result["y"][1]
    assert result["y"][1] == pytest.approx(0.0, abs=0.1)


def test_screen_multiple_observables(continuous_factors: list[Factor]) -> None:
    def multi_obs(cfg: dict[str, Any]) -> dict[str, float]:
        return {
            "obs1": cfg["alpha"],
            "obs2": cfg["beta"],
        }

    result = screen(multi_obs, continuous_factors, n_trajectories=20, seed=0)
    assert set(result.keys()) == {"obs1", "obs2"}
    assert result["obs1"].shape == (2,)
    assert result["obs2"].shape == (2,)


def test_screen_rejects_non_morris() -> None:
    factors = [Factor("x", FactorType.CONTINUOUS, bounds=(0.0, 1.0))]
    with pytest.raises(NotImplementedError, match="not yet implemented"):
        screen(lambda _c: {"y": 0.0}, factors, method="sobol")


def test_screen_rejects_no_continuous() -> None:
    factors = [Factor("m", FactorType.CATEGORICAL, levels=["a", "b"])]
    with pytest.raises(ValueError, match="at least one continuous"):
        screen(lambda _c: {"y": 0.0}, factors)


# ---------------------------------------------------------------------------
# reduce_factors (#10)
# ---------------------------------------------------------------------------


def test_reduce_keeps_influential(continuous_factors: list[Factor]) -> None:
    importance = {"y": np.array([0.5, 0.01])}
    kept = reduce_factors(continuous_factors, importance, threshold=0.1)
    names = [f.name for f in kept]
    assert "alpha" in names
    assert "beta" not in names


def test_reduce_always_keeps_non_continuous() -> None:
    factors = [
        Factor("method", FactorType.CATEGORICAL, levels=["a", "b"]),
        Factor("alpha", FactorType.CONTINUOUS, bounds=(0.0, 1.0)),
    ]
    importance = {"y": np.array([0.001])}  # alpha unimportant
    kept = reduce_factors(factors, importance, threshold=0.1)
    names = [f.name for f in kept]
    assert "method" in names
    assert "alpha" not in names


def test_reduce_multiple_observables(continuous_factors: list[Factor]) -> None:
    importance = {
        "obs1": np.array([0.05, 0.3]),
        "obs2": np.array([0.2, 0.01]),
    }
    kept = reduce_factors(continuous_factors, importance, threshold=0.1)
    names = {f.name for f in kept}
    # alpha important in obs2, beta important in obs1 → both kept
    assert names == {"alpha", "beta"}


def test_reduce_threshold_zero_keeps_all(continuous_factors: list[Factor]) -> None:
    importance = {"y": np.array([0.0, 0.0])}
    kept = reduce_factors(continuous_factors, importance, threshold=0.0)
    assert len(kept) == 2


def test_reduce_high_threshold_drops_all(continuous_factors: list[Factor]) -> None:
    importance = {"y": np.array([0.1, 0.1])}
    kept = reduce_factors(continuous_factors, importance, threshold=0.5)
    assert len(kept) == 0
