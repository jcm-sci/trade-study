"""Model stacking and ensemble weights.

Bayesian stacking via arviz (for models with log-likelihoods) and
score-based stacking via scipy (for arbitrary score matrices).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def stack_bayesian(
    compare_dict: dict[str, Any],
    *,
    method: str = "stacking",
) -> dict[str, float]:
    """Bayesian stacking via arviz.compare.

    Args:
        compare_dict: Dictionary mapping model names to arviz DataTree
            or ELPDData objects (must contain log_likelihood group).
        method: Weighting method. One of "stacking", "BB-pseudo-BMA",
            "pseudo-BMA".

    Returns:
        Dictionary mapping model names to stacking weights.
    """
    import arviz as az  # type: ignore[import-untyped]

    result = az.compare(compare_dict, method=method)
    return dict(zip(result.index, result["weight"], strict=True))


def stack_scores(
    score_matrix: NDArray[np.floating[Any]],
    *,
    maximize: bool = False,
) -> NDArray[np.floating[Any]]:
    """Optimize stacking weights from a score matrix.

    For non-Bayesian models where log-likelihoods aren't available.
    Finds weights w on the simplex that optimize the weighted composite score.

    Args:
        score_matrix: Array of shape (n_models, n_test_points) where each
            entry is the score of model i on test point j.
        maximize: If True, maximize the weighted score; if False, minimize.

    Returns:
        Array of weights, shape (n_models,), summing to 1.
    """
    from scipy.optimize import minimize  # type: ignore[import-untyped]

    n_models = score_matrix.shape[0]

    def objective(w: NDArray[np.floating[Any]]) -> float:
        composite = w @ score_matrix
        val = float(np.mean(composite))
        return -val if maximize else val

    # Simplex constraint: weights >= 0 and sum to 1
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    bounds = [(0.0, 1.0)] * n_models
    w0 = np.ones(n_models) / n_models

    result = minimize(
        objective,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    return np.asarray(result.x, dtype=np.float64)


def stack_proportional(
    score_matrix: NDArray[np.floating[Any]],
    *,
    maximize: bool = False,
) -> NDArray[np.floating[Any]]:
    """Weight models in direct proportion to their mean score.

    Unlike :func:`stack_scores` (a linear program that puts *all* weight
    on the single best-performing model whenever there's any nonzero gap,
    even a noise-scale one), this scales smoothly with relative
    performance -- useful when "how much better" should matter, not just
    "which one is best". Two near-tied models get near-equal weights
    here; under :func:`stack_scores` the same tiny gap can flip the
    result between an even split and 100% on one model, since a linear
    objective over a simplex has no reason to split weight once any
    model is even infinitesimally ahead.

    Args:
        score_matrix: Array of shape (n_models, n_test_points) where each
            entry is the score of model i on test point j.
        maximize: If True, higher scores are better. If False, lower
            scores are better; scores are inverted internally
            (``max - score``) rather than divided, so a score of exactly
            0 doesn't produce an unbounded weight.

    Returns:
        Array of weights, shape (n_models,), summing to 1, each >= 0.
        Falls back to a uniform split if every model scores identically
        (nothing to distinguish them by).
    """
    mean_scores = np.mean(score_matrix, axis=1)
    if not maximize:
        mean_scores = np.max(mean_scores) - mean_scores
    mean_scores = np.clip(mean_scores, a_min=0.0, a_max=None)

    total = mean_scores.sum()
    n_models = score_matrix.shape[0]
    if total <= 0.0:
        return np.full(n_models, 1.0 / n_models, dtype=np.float64)
    return np.asarray(mean_scores / total, dtype=np.float64)


def ensemble_predict(
    predictions: list[NDArray[np.floating[Any]]],
    weights: NDArray[np.floating[Any]],
) -> NDArray[np.floating[Any]]:
    """Weighted ensemble of model predictions.

    Args:
        predictions: List of prediction arrays, one per model.
            Each should have the same shape.
        weights: Stacking weights, shape (n_models,).

    Returns:
        Weighted average prediction array.
    """
    w = np.asarray(weights, dtype=np.float64)
    w /= w.sum()
    result = np.zeros_like(predictions[0], dtype=np.float64)
    for pred, wi in zip(predictions, w, strict=True):
        result += wi * pred
    return result
