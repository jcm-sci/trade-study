"""Tests for coverage_curve (issue #6)."""

from __future__ import annotations

import numpy as np
import pytest

from trade_study.scoring import coverage_curve

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
