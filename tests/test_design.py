"""Tests for design module (issues #8, #9, #10)."""

from __future__ import annotations

import importlib
import warnings
from typing import Any

import numpy as np
import pytest

from trade_study.design import (
    Factor,
    FactorConstraint,
    FactorType,
    build_grid,
    reduce_factors,
    screen,
    sobol_indices,
)

_design = importlib.import_module("trade_study.design")

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


def test_factor_empty_name() -> None:
    """Empty factor name is rejected."""
    with pytest.raises(ValueError, match="non-empty string"):
        Factor("", FactorType.CATEGORICAL, levels=["a"])


def test_factor_empty_levels() -> None:
    """Empty levels list is rejected for categorical and discrete."""
    with pytest.raises(ValueError, match="non-empty"):
        Factor("x", FactorType.CATEGORICAL, levels=[])
    with pytest.raises(ValueError, match="non-empty"):
        Factor("x", FactorType.DISCRETE, levels=[])


def test_factor_inverted_bounds() -> None:
    """Inverted or equal bounds are rejected for continuous factors."""
    with pytest.raises(ValueError, match="lo < hi"):
        Factor("x", FactorType.CONTINUOUS, bounds=(10.0, 1.0))
    with pytest.raises(ValueError, match="lo < hi"):
        Factor("x", FactorType.CONTINUOUS, bounds=(5.0, 5.0))


def test_factor_nonfinite_bounds() -> None:
    """NaN and inf bounds are rejected for continuous factors."""
    with pytest.raises(ValueError, match="finite"):
        Factor("x", FactorType.CONTINUOUS, bounds=(float("nan"), 1.0))
    with pytest.raises(ValueError, match="finite"):
        Factor("x", FactorType.CONTINUOUS, bounds=(0.0, float("inf")))


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
# build_grid — QMC: Sobol & Halton (#44)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("qmc_method", ["sobol", "halton"])
def test_qmc_sample_count(
    continuous_factors: list[Factor],
    qmc_method: str,
) -> None:
    grid = build_grid(continuous_factors, method=qmc_method, n_samples=64)
    assert len(grid) == 64


@pytest.mark.parametrize("qmc_method", ["sobol", "halton"])
def test_qmc_continuous_bounds(
    continuous_factors: list[Factor],
    qmc_method: str,
) -> None:
    grid = build_grid(continuous_factors, method=qmc_method, n_samples=128)
    alphas = [cfg["alpha"] for cfg in grid]
    betas = [cfg["beta"] for cfg in grid]
    assert all(0.0 <= a <= 1.0 for a in alphas)
    assert all(10.0 <= b <= 20.0 for b in betas)


@pytest.mark.parametrize("qmc_method", ["sobol", "halton"])
def test_qmc_categorical_in_levels(
    mixed_factors: list[Factor],
    qmc_method: str,
) -> None:
    grid = build_grid(mixed_factors, method=qmc_method, n_samples=64)
    colours = {cfg["colour"] for cfg in grid}
    assert colours <= {"red", "green", "blue"}


@pytest.mark.parametrize("qmc_method", ["sobol", "halton"])
def test_qmc_deterministic_with_seed(
    continuous_factors: list[Factor],
    qmc_method: str,
) -> None:
    g1 = build_grid(continuous_factors, method=qmc_method, n_samples=32, seed=7)
    g2 = build_grid(continuous_factors, method=qmc_method, n_samples=32, seed=7)
    assert g1 == g2


@pytest.mark.parametrize("qmc_method", ["sobol", "halton"])
def test_qmc_different_seeds_differ(
    continuous_factors: list[Factor],
    qmc_method: str,
) -> None:
    g1 = build_grid(continuous_factors, method=qmc_method, n_samples=32, seed=1)
    g2 = build_grid(continuous_factors, method=qmc_method, n_samples=32, seed=2)
    assert g1 != g2


def test_qmc_scramble_off(continuous_factors: list[Factor]) -> None:
    g1 = build_grid(
        continuous_factors,
        method="sobol",
        n_samples=32,
        scramble=False,
    )
    g2 = build_grid(
        continuous_factors,
        method="sobol",
        n_samples=32,
        scramble=False,
    )
    assert g1 == g2


def test_qmc_sobol_vs_halton_differ(continuous_factors: list[Factor]) -> None:
    g_sobol = build_grid(
        continuous_factors,
        method="sobol",
        n_samples=32,
        seed=0,
        scramble=False,
    )
    g_halton = build_grid(
        continuous_factors,
        method="halton",
        n_samples=32,
        seed=0,
        scramble=False,
    )
    assert g_sobol != g_halton


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


def test_screen_rejects_unknown_method() -> None:
    factors = [Factor("x", FactorType.CONTINUOUS, bounds=(0.0, 1.0))]
    with pytest.raises(ValueError, match="Unknown screening method"):
        screen(lambda _c: {"y": 0.0}, factors, method="bogus")


def test_screen_rejects_no_continuous() -> None:
    factors = [Factor("m", FactorType.CATEGORICAL, levels=["a", "b"])]
    with pytest.raises(ValueError, match="at least one continuous"):
        screen(lambda _c: {"y": 0.0}, factors)


# ---------------------------------------------------------------------------
# screen/sobol_indices — replicate averaging (#122)
# ---------------------------------------------------------------------------


def test_run_fn_accepts_rep_detects_rep_param() -> None:
    def with_rep(_c: dict[str, Any], *, rep: int = 0) -> dict[str, float]:
        return {"y": float(rep)}

    def without_rep(_c: dict[str, Any]) -> dict[str, float]:
        return {"y": 0.0}

    assert _design._run_fn_accepts_rep(with_rep) is True  # ruff: ignore[private-member-access]
    assert _design._run_fn_accepts_rep(without_rep) is False  # ruff: ignore[private-member-access]


def test_evaluate_averaged_single_call_when_n_reps_one() -> None:
    calls: list[int] = []

    def run_fn(_c: dict[str, Any]) -> dict[str, float]:
        calls.append(1)
        return {"y": 5.0}

    result = _design._evaluate_averaged(run_fn, {}, 1, supports_rep=False)  # ruff: ignore[private-member-access]
    assert result == {"y": 5.0}
    assert len(calls) == 1


def test_evaluate_averaged_averages_across_reps() -> None:
    def run_fn(_c: dict[str, Any], *, rep: int = 0) -> dict[str, float]:
        return {"y": float(rep)}

    # average of reps 0..3 is 1.5
    result = _design._evaluate_averaged(run_fn, {}, 4, supports_rep=True)  # ruff: ignore[private-member-access]
    assert result == pytest.approx({"y": 1.5})


def test_evaluate_averaged_repeats_identical_draw_when_run_fn_ignores_rep() -> None:
    calls: list[int] = []

    def run_fn(_c: dict[str, Any]) -> dict[str, float]:
        calls.append(1)
        return {"y": 7.0}

    result = _design._evaluate_averaged(run_fn, {}, 3, supports_rep=False)  # ruff: ignore[private-member-access]
    assert result == pytest.approx({"y": 7.0})
    assert len(calls) == 3


def test_screen_rejects_invalid_n_reps(continuous_factors: list[Factor]) -> None:
    with pytest.raises(ValueError, match="n_reps must be >= 1"):
        screen(_linear_model, continuous_factors, n_trajectories=5, n_reps=0)


def test_screen_n_reps_passes_incrementing_rep_to_run_fn(
    continuous_factors: list[Factor],
) -> None:
    seen_reps: set[int] = set()

    def run_fn(cfg: dict[str, Any], *, rep: int = 0) -> dict[str, float]:
        seen_reps.add(rep)
        return {"y": cfg["alpha"]}

    screen(run_fn, continuous_factors, n_trajectories=5, n_reps=3, seed=0)
    assert seen_reps == {0, 1, 2}


def test_sobol_indices_rejects_invalid_n_reps(continuous_factors: list[Factor]) -> None:
    with pytest.raises(ValueError, match="n_reps must be >= 1"):
        sobol_indices(_linear_model, continuous_factors, n_samples=8, n_reps=0)


def test_sobol_indices_n_reps_passes_incrementing_rep_to_run_fn(
    continuous_factors: list[Factor],
) -> None:
    seen_reps: set[int] = set()

    def run_fn(cfg: dict[str, Any], *, rep: int = 0) -> dict[str, float]:
        seen_reps.add(rep)
        return {"y": cfg["alpha"]}

    sobol_indices(run_fn, continuous_factors, n_samples=8, n_reps=3, seed=0)
    assert seen_reps == {0, 1, 2}


# ---------------------------------------------------------------------------
# screen — Sobol (#76)
# ---------------------------------------------------------------------------


def test_screen_sobol_returns_dict(continuous_factors: list[Factor]) -> None:
    result = screen(
        _linear_model,
        continuous_factors,
        method="sobol",
        n_trajectories=64,
        seed=0,
    )
    assert isinstance(result, dict)
    assert "y" in result


def test_screen_sobol_importance_shape(continuous_factors: list[Factor]) -> None:
    result = screen(
        _linear_model,
        continuous_factors,
        method="sobol",
        n_trajectories=64,
        seed=0,
    )
    assert result["y"].shape == (2,)


def test_screen_sobol_detects_influential_factor(
    continuous_factors: list[Factor],
) -> None:
    """Sobol S1 for alpha should dominate; beta should be near zero."""
    result = screen(
        _linear_model,
        continuous_factors,
        method="sobol",
        n_trajectories=256,
        seed=0,
    )
    assert result["y"][0] > result["y"][1]
    assert result["y"][1] == pytest.approx(0.0, abs=0.1)


def test_screen_sobol_multiple_observables(
    continuous_factors: list[Factor],
) -> None:
    def multi_obs(cfg: dict[str, Any]) -> dict[str, float]:
        return {
            "obs1": cfg["alpha"],
            "obs2": cfg["beta"],
        }

    result = screen(
        multi_obs,
        continuous_factors,
        method="sobol",
        n_trajectories=64,
        seed=0,
    )
    assert set(result.keys()) == {"obs1", "obs2"}
    assert result["obs1"].shape == (2,)
    assert result["obs2"].shape == (2,)


# ---------------------------------------------------------------------------
# sobol_indices (#120)
# ---------------------------------------------------------------------------


def test_sobol_indices_returns_tuple_per_observable(
    continuous_factors: list[Factor],
) -> None:
    result = sobol_indices(_linear_model, continuous_factors, n_samples=64, seed=0)
    assert isinstance(result, dict)
    assert "y" in result
    s1, st = result["y"]
    assert s1.shape == (2,)
    assert st.shape == (2,)


def test_sobol_indices_matches_screen_s1(continuous_factors: list[Factor]) -> None:
    """S1 from sobol_indices should match screen(method="sobol")'s S1."""
    screened = screen(
        _linear_model, continuous_factors, method="sobol", n_trajectories=64, seed=0
    )
    result = sobol_indices(_linear_model, continuous_factors, n_samples=64, seed=0)
    s1, _st = result["y"]
    np.testing.assert_allclose(s1, screened["y"])


def test_sobol_indices_total_order_at_least_first_order(
    continuous_factors: list[Factor],
) -> None:
    """ST >= S1 holds (up to MC estimation noise) for a variance decomposition."""
    result = sobol_indices(_linear_model, continuous_factors, n_samples=512, seed=0)
    s1, st = result["y"]
    # SALib's S1/ST are independently-estimated MC quantities, not computed
    # from a shared exact decomposition, so a generous tolerance is needed
    # even at a reasonable sample size -- this only guards against a
    # systematic ST-vs-S1 mixup (e.g. an accidental swap), not tight
    # numerical agreement.
    assert np.all(st >= s1 - 0.05)


def test_sobol_indices_detects_pure_interaction(
    continuous_factors: list[Factor],
) -> None:
    """A pure product of two zero-mean factors has ~zero S1 but nonzero ST."""

    def interaction_model(cfg: dict[str, Any]) -> dict[str, float]:
        # Both factors centered to zero mean: a pure product term then has
        # zero first-order effect for *both* factors (E[Y|A] = A*E[B] = 0
        # and vice versa), so any measured S1 is interaction leaking in,
        # while ST captures the interaction directly.
        alpha_centered = (cfg["alpha"] - 0.5) / 0.5  # [0, 1] -> [-1, 1]
        beta_centered = (cfg["beta"] - 15.0) / 5.0  # [10, 20] -> [-1, 1]
        return {"y": alpha_centered * beta_centered}

    result = sobol_indices(interaction_model, continuous_factors, n_samples=256, seed=0)
    s1, st = result["y"]
    assert np.all(s1 < 0.1)
    assert np.all(st > 0.1)


def test_sobol_indices_rejects_no_continuous() -> None:
    factors = [Factor("m", FactorType.CATEGORICAL, levels=["a", "b"])]
    with pytest.raises(ValueError, match="at least one continuous"):
        sobol_indices(lambda _c: {"y": 0.0}, factors)


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


def test_reduce_nan_in_one_observable_does_not_erase_another(
    continuous_factors: list[Factor],
) -> None:
    """A NaN-valued observable (#119) can't erase a real signal found elsewhere.

    alpha has a real, large importance (0.5) on obs1; obs2 -- a
    conditionally-undefined observable (e.g. a Type-I rate) -- is NaN for
    every factor. Before #119's fix, np.maximum propagated that NaN and
    silently dropped alpha despite obs1's real signal.
    """
    importance = {
        "obs1": np.array([0.5, 0.01]),
        "obs2": np.array([np.nan, np.nan]),
    }
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        kept = reduce_factors(continuous_factors, importance, threshold=0.1)
    names = [f.name for f in kept]
    assert "alpha" in names
    assert "beta" not in names


def test_reduce_all_nan_for_factor_drops_and_warns(
    continuous_factors: list[Factor],
) -> None:
    """A factor NaN across every observable is dropped, but with a warning."""
    importance = {
        "obs1": np.array([np.nan, 0.5]),
        "obs2": np.array([np.nan, 0.01]),
    }
    with pytest.warns(UserWarning, match="alpha"):
        kept = reduce_factors(continuous_factors, importance, threshold=0.1)
    names = [f.name for f in kept]
    assert "alpha" not in names
    assert "beta" in names


def test_reduce_no_nan_does_not_warn(continuous_factors: list[Factor]) -> None:
    importance = {"y": np.array([0.5, 0.01])}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        reduce_factors(continuous_factors, importance, threshold=0.1)


# ---------------------------------------------------------------------------
# FactorConstraint — coupled design-time constraints (#103)
# ---------------------------------------------------------------------------


@pytest.fixture
def coupled_factors() -> list[Factor]:
    """Two coupled categorical factors used in constraint tests.

    Returns:
        ``method`` in {``elbo_only``, ``mixed``, ``full``} and
        ``patience`` in {1, 2, 3, 5, 10}.
    """
    return [
        Factor(
            "method",
            FactorType.CATEGORICAL,
            levels=["elbo_only", "mixed", "full"],
        ),
        Factor("patience", FactorType.DISCRETE, levels=[1, 2, 3, 5, 10]),
    ]


def _elbo_short_patience(cfg: dict[str, Any]) -> bool:
    """Toy coupled rule used across constraint tests.

    Args:
        cfg: Candidate config dict.

    Returns:
        ``True`` unless ``method == "elbo_only"`` with ``patience > 2``.
    """
    return cfg["method"] != "elbo_only" or cfg["patience"] <= 2


def test_factor_constraint_is_feasible() -> None:
    c = FactorConstraint(_elbo_short_patience, name="elbo_short_patience")
    assert c.is_feasible({"method": "elbo_only", "patience": 1})
    assert not c.is_feasible({"method": "elbo_only", "patience": 5})
    assert c.is_feasible({"method": "mixed", "patience": 10})


def test_full_factorial_filters_constraints(coupled_factors: list[Factor]) -> None:
    grid = build_grid(
        coupled_factors,
        method="full",
        constraints=[FactorConstraint(_elbo_short_patience)],
    )
    # 3 * 5 = 15 unconstrained; reject elbo_only with patience in {3, 5, 10}.
    assert len(grid) == 15 - 3
    for cfg in grid:
        assert _elbo_short_patience(cfg)


def test_full_factorial_no_constraints_unchanged(
    coupled_factors: list[Factor],
) -> None:
    base = build_grid(coupled_factors, method="full")
    same = build_grid(coupled_factors, method="full", constraints=[])
    assert base == same


def test_full_factorial_all_rejected_returns_empty(
    coupled_factors: list[Factor],
) -> None:
    grid = build_grid(
        coupled_factors,
        method="full",
        constraints=[FactorConstraint(lambda _c: False, name="reject_all")],
    )
    assert grid == []


@pytest.mark.parametrize("method", ["sobol", "halton", "lhs"])
def test_qmc_lhs_returns_n_feasible(
    coupled_factors: list[Factor],
    method: str,
) -> None:
    n = 32
    grid = build_grid(
        coupled_factors,
        method=method,
        n_samples=n,
        seed=0,
        constraints=[FactorConstraint(_elbo_short_patience)],
    )
    assert len(grid) == n
    for cfg in grid:
        assert _elbo_short_patience(cfg)


@pytest.mark.parametrize("method", ["sobol", "halton", "lhs"])
def test_qmc_lhs_constrained_deterministic(
    coupled_factors: list[Factor],
    method: str,
) -> None:
    constraints = [FactorConstraint(_elbo_short_patience)]
    g1 = build_grid(
        coupled_factors,
        method=method,
        n_samples=16,
        seed=11,
        constraints=constraints,
    )
    g2 = build_grid(
        coupled_factors,
        method=method,
        n_samples=16,
        seed=11,
        constraints=constraints,
    )
    assert g1 == g2


def test_qmc_constraints_raise_on_infeasible(
    coupled_factors: list[Factor],
) -> None:
    with pytest.raises(ValueError, match="rejected too many"):
        build_grid(
            coupled_factors,
            method="sobol",
            n_samples=8,
            seed=0,
            constraints=[FactorConstraint(lambda _c: False, name="reject_all")],
        )


def test_qmc_low_feasibility_warns() -> None:
    """A very tight constraint should still succeed but warn."""
    factors = [
        Factor("x", FactorType.CONTINUOUS, bounds=(0.0, 1.0)),
    ]
    # Accept only configs in the bottom 5% of x's range — feasibility ~0.05.
    constraints = [FactorConstraint(lambda c: c["x"] < 0.05, name="tight")]
    with pytest.warns(UserWarning, match="low feasibility ratio"):
        grid = build_grid(
            factors,
            method="sobol",
            n_samples=4,
            seed=0,
            constraints=constraints,
        )
    assert len(grid) == 4
    assert all(cfg["x"] < 0.05 for cfg in grid)
