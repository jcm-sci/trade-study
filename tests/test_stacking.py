"""Tests for stacking module (issues #15, #16, #17)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from trade_study.stacking import ensemble_predict, stack_bayesian, stack_scores

RNG = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# stack_bayesian (#15)
# ---------------------------------------------------------------------------


def _make_arviz_datatree(
    log_lik: np.ndarray,
) -> Any:
    """Build a minimal arviz DataTree with posterior and log_likelihood groups.

    Args:
        log_lik: Array of shape (chains, draws, n_obs).

    Returns:
        An arviz DataTree with posterior and log_likelihood.
    """
    import arviz as az  # type: ignore[import-untyped]

    n_chains, n_draws, _n_obs = log_lik.shape
    posterior = {"mu": RNG.standard_normal((n_chains, n_draws))}
    return az.from_dict({
        "posterior": posterior,
        "log_likelihood": {"obs": log_lik},
    })


def test_stack_bayesian_weights_sum_to_one() -> None:
    n_obs = 50
    # Model A: better log-likelihoods (closer to 0)
    ll_a = RNG.normal(-1.0, 0.5, size=(1, 200, n_obs))
    # Model B: worse log-likelihoods
    ll_b = RNG.normal(-5.0, 0.5, size=(1, 200, n_obs))
    compare_dict = {
        "model_a": _make_arviz_datatree(ll_a),
        "model_b": _make_arviz_datatree(ll_b),
    }
    weights = stack_bayesian(compare_dict)
    assert sum(weights.values()) == pytest.approx(1.0)


def test_stack_bayesian_best_model_highest_weight() -> None:
    n_obs = 50
    ll_good = RNG.normal(-0.5, 0.3, size=(1, 200, n_obs))
    ll_bad = RNG.normal(-10.0, 0.3, size=(1, 200, n_obs))
    compare_dict = {
        "good": _make_arviz_datatree(ll_good),
        "bad": _make_arviz_datatree(ll_bad),
    }
    weights = stack_bayesian(compare_dict)
    assert weights["good"] > weights["bad"]


def test_stack_bayesian_returns_all_models() -> None:
    n_obs = 30
    compare_dict = {
        f"m{i}": _make_arviz_datatree(RNG.normal(-2.0, 1.0, size=(1, 100, n_obs)))
        for i in range(3)
    }
    weights = stack_bayesian(compare_dict)
    assert set(weights.keys()) == {"m0", "m1", "m2"}


# ---------------------------------------------------------------------------
# stack_scores (#16)
# ---------------------------------------------------------------------------


def test_stack_scores_weights_sum_to_one() -> None:
    scores = RNG.standard_normal((3, 50))
    weights = stack_scores(scores)
    assert np.sum(weights) == pytest.approx(1.0)


def test_stack_scores_weights_non_negative() -> None:
    scores = RNG.standard_normal((3, 50))
    weights = stack_scores(scores)
    assert np.all(weights >= -1e-10)


def test_stack_scores_dominant_model_minimize() -> None:
    """Model 0 has lowest scores everywhere; minimize -> weight on model 0."""
    scores = np.array([
        [0.1, 0.2, 0.1, 0.15],  # best (lowest)
        [5.0, 6.0, 5.5, 5.2],
        [9.0, 8.0, 9.5, 8.8],
    ])
    weights = stack_scores(scores, maximize=False)
    assert weights[0] > 0.9


def test_stack_scores_dominant_model_maximize() -> None:
    """Model 0 has highest scores; maximize -> weight on model 0."""
    scores = np.array([
        [9.0, 8.5, 9.2, 8.8],  # best (highest)
        [1.0, 1.5, 1.2, 1.1],
        [0.1, 0.2, 0.15, 0.1],
    ])
    weights = stack_scores(scores, maximize=True)
    assert weights[0] > 0.9


def test_stack_scores_shape() -> None:
    scores = RNG.standard_normal((4, 20))
    weights = stack_scores(scores)
    assert weights.shape == (4,)


def test_stack_scores_dtype() -> None:
    scores = RNG.standard_normal((2, 10))
    weights = stack_scores(scores)
    assert weights.dtype == np.float64


# ---------------------------------------------------------------------------
# ensemble_predict (#17)
# ---------------------------------------------------------------------------


def test_ensemble_equal_weights_is_mean() -> None:
    p1 = np.array([1.0, 2.0, 3.0])
    p2 = np.array([3.0, 4.0, 5.0])
    weights = np.array([1.0, 1.0])
    result = ensemble_predict([p1, p2], weights)
    expected = (p1 + p2) / 2.0
    np.testing.assert_allclose(result, expected)


def test_ensemble_single_weight_one_is_identity() -> None:
    p1 = np.array([10.0, 20.0])
    weights = np.array([1.0])
    result = ensemble_predict([p1], weights)
    np.testing.assert_allclose(result, p1)


def test_ensemble_weighted_sum() -> None:
    p1 = np.array([0.0, 0.0])
    p2 = np.array([10.0, 10.0])
    weights = np.array([0.25, 0.75])
    result = ensemble_predict([p1, p2], weights)
    expected = np.array([7.5, 7.5])
    np.testing.assert_allclose(result, expected)


def test_ensemble_2d_predictions() -> None:
    p1 = np.ones((3, 4))
    p2 = np.full((3, 4), 3.0)
    weights = np.array([0.5, 0.5])
    result = ensemble_predict([p1, p2], weights)
    np.testing.assert_allclose(result, np.full((3, 4), 2.0))


def test_ensemble_dtype() -> None:
    p1 = np.array([1.0, 2.0])
    weights = np.array([1.0])
    result = ensemble_predict([p1], weights)
    assert result.dtype == np.float64


def test_ensemble_weights_normalised() -> None:
    """Weights that don't sum to 1 get normalised internally."""
    p1 = np.array([2.0, 4.0])
    p2 = np.array([6.0, 8.0])
    weights = np.array([2.0, 2.0])  # sum=4, should be treated as 0.5/0.5
    result = ensemble_predict([p1, p2], weights)
    expected = (p1 + p2) / 2.0
    np.testing.assert_allclose(result, expected)
