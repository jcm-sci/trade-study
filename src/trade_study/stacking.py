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
    temperature: float = 1.0,
) -> NDArray[np.floating[Any]]:
    """Weight models proportionally to exponential score utilities.

    Mean scores are transformed using a direction-aware softmax. Near-tied
    models receive comparable weights in either direction, including when
    scores are zero or negative. These are heuristic ensemble weights,
    rather than Bayesian stacking weights or calibrated probabilities.

    Args:
        score_matrix: Finite array of shape (n_models, n_test_points).
        maximize: Whether higher scores are better. Default is minimize.
        temperature: Positive, finite scale in score units. Larger values
            spread weight more evenly; smaller values favor the best model.
            The default is 1.0. Scale this with the scores when changing
            their units to preserve the same weights.

    Returns:
        Non-negative float64 weights summing to 1, shape (n_models,).
        Exactly equal mean scores receive uniform weights.

    Raises:
        ValueError: If scores are empty, non-finite, or not two-dimensional,
            or temperature is not positive and finite.
    """
    scores = np.asarray(score_matrix, dtype=np.float64)
    if scores.ndim != 2 or 0 in scores.shape or not np.all(np.isfinite(scores)):
        msg = "score_matrix must be a non-empty, finite two-dimensional array"
        raise ValueError(msg)
    if not np.isfinite(temperature) or temperature <= 0:
        msg = "temperature must be positive and finite"
        raise ValueError(msg)

    # Divide before summing to avoid overflowing the mean of large scores.
    means = np.sum(scores / scores.shape[1], axis=1)
    utility = means if maximize else -means
    # Overflow in a very large unfavorable gap means a zero softmax weight.
    with np.errstate(over="ignore"):
        logits = (utility - np.max(utility)) / temperature
    weights = np.exp(logits)
    return np.asarray(weights / weights.sum(), dtype=np.float64)


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
