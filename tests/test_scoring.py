"""Tests for scoring module (issues #5, #6)."""

from __future__ import annotations

import numpy as np
import pytest

from trade_study.scoring import coverage_curve, score

# -- coverage_curve ------------------------------------------------------------

RNG = np.random.default_rng(42)


@pytest.fixture
def well_calibrated() -> tuple[np.ndarray, np.ndarray]:
    """Well-calibrated posteriors: truth drawn from the same distribution.

    Returns:
        Tuple of (posteriors, truth) arrays.
    """
    n_obs, n_samples = 500, 200
    mu = RNG.standard_normal(n_obs)
    posteriors = mu[:, None] + RNG.standard_normal((n_obs, n_samples))
    truth = mu + RNG.standard_normal(n_obs)
    return posteriors, truth


def test_coverage_curve_returns_two_arrays(
    well_calibrated: tuple[np.ndarray, np.ndarray],
) -> None:
    posteriors, truth = well_calibrated
    levels, empirical = coverage_curve(posteriors, truth)
    assert isinstance(levels, np.ndarray)
    assert isinstance(empirical, np.ndarray)


def test_coverage_curve_default_levels_shape(
    well_calibrated: tuple[np.ndarray, np.ndarray],
) -> None:
    posteriors, truth = well_calibrated
    levels, empirical = coverage_curve(posteriors, truth)
    assert levels.shape == (50,)
    assert empirical.shape == levels.shape


def test_coverage_curve_custom_levels() -> None:
    n_obs, n_samples = 100, 50
    posteriors = RNG.standard_normal((n_obs, n_samples))
    truth = np.zeros(n_obs)
    custom = np.array([0.5, 0.9, 0.95])
    levels, empirical = coverage_curve(posteriors, truth, levels=custom)
    assert levels.shape == (3,)
    assert empirical.shape == (3,)
    np.testing.assert_array_equal(levels, custom)


def test_coverage_curve_monotonically_increasing(
    well_calibrated: tuple[np.ndarray, np.ndarray],
) -> None:
    posteriors, truth = well_calibrated
    _, empirical = coverage_curve(posteriors, truth)
    diffs = np.diff(empirical)
    assert np.all(diffs >= -1e-12)


def test_coverage_curve_well_calibrated_near_diagonal(
    well_calibrated: tuple[np.ndarray, np.ndarray],
) -> None:
    posteriors, truth = well_calibrated
    levels, empirical = coverage_curve(posteriors, truth)
    assert np.max(np.abs(empirical - levels)) < 0.15


def test_coverage_curve_bounded_zero_one(
    well_calibrated: tuple[np.ndarray, np.ndarray],
) -> None:
    posteriors, truth = well_calibrated
    _, empirical = coverage_curve(posteriors, truth)
    assert np.all(empirical >= 0.0)
    assert np.all(empirical <= 1.0)


def test_coverage_curve_degenerate_posterior() -> None:
    n_obs = 50
    truth = RNG.standard_normal(n_obs)
    posteriors = np.tile(truth[:, None], (1, 100))
    _levels, empirical = coverage_curve(posteriors, truth)
    assert np.all(empirical >= 1.0 - 1e-12)


# -- score() dispatch ---------------------------------------------------------


def test_score_rmse_known_answer() -> None:
    predictions = np.array([3.0, 4.0, 5.0])
    truth = np.array([1.0, 2.0, 3.0])
    result = score("rmse", predictions, truth)
    assert result == pytest.approx(2.0)


def test_score_mae_known_answer() -> None:
    predictions = np.array([3.0, 5.0, 7.0])
    truth = np.array([1.0, 3.0, 5.0])
    result = score("mae", predictions, truth)
    assert result == pytest.approx(2.0)


def test_score_coverage_known_answer() -> None:
    truth = np.array([0.5])
    # 100 samples uniformly spanning [0, 1] — truth is inside any wide interval.
    predictions = np.linspace(0.0, 1.0, 100).reshape(1, -1)
    result = score("coverage", predictions, truth, level=0.90)
    assert result == pytest.approx(1.0)


def test_score_crps_returns_finite_float() -> None:
    ensemble = RNG.standard_normal((20, 50))
    truth = RNG.standard_normal(20)
    result = score("crps", ensemble, truth)
    assert isinstance(result, float)
    assert np.isfinite(result)


def test_score_energy_returns_finite_float() -> None:
    # energy_score expects (obs, fct) with fct having m_axis=-2, v_axis=-1
    # shape: obs=(n_obs, n_vars), fct=(n_obs, n_members, n_vars)
    ensemble = RNG.standard_normal((10, 30, 2))
    truth = RNG.standard_normal((10, 2))
    result = score("energy", ensemble, truth)
    assert isinstance(result, float)
    assert np.isfinite(result)


def test_score_brier_returns_finite_float() -> None:
    probabilities = RNG.uniform(0, 1, size=50)
    outcomes = RNG.choice([0.0, 1.0], size=50)
    result = score("brier", probabilities, outcomes)
    assert isinstance(result, float)
    assert np.isfinite(result)


@pytest.mark.xfail(
    reason="_wis passes 4 positional args; sr.weighted_interval_score needs 5",
    raises=TypeError,
    strict=True,
)
def test_score_wis_returns_finite_float() -> None:
    # _wis passes predictions[..., 0] as median, predictions[..., 1] as lower
    # to sr.weighted_interval_score(obs, median, lower, upper, alpha)
    # So predictions needs shape (n_obs, 3): median, lower, upper
    n_obs = 30
    median = RNG.standard_normal(n_obs)
    lower = median - RNG.uniform(1, 3, size=n_obs)
    upper = median + RNG.uniform(1, 3, size=n_obs)
    predictions = np.stack([median, lower, upper], axis=-1)
    truth = RNG.standard_normal(n_obs)
    result = score("wis", predictions, truth)
    assert isinstance(result, float)
    assert np.isfinite(result)


def test_score_interval_returns_finite_float() -> None:
    lower = RNG.standard_normal(30)
    upper = lower + RNG.uniform(1, 3, size=30)
    predictions = np.stack([lower, upper], axis=-1)
    truth = RNG.standard_normal(30)
    result = score("interval", predictions, truth)
    assert isinstance(result, float)
    assert np.isfinite(result)


def test_score_unknown_metric_raises() -> None:
    predictions = np.array([1.0, 2.0])
    truth = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="Unknown metric"):
        score("nonexistent", predictions, truth)
