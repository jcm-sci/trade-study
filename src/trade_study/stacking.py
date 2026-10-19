"""Model stacking and ensemble weights.

Bayesian stacking via arviz (for models with log-likelihoods) and
NumPy score-based selection and ensemble prediction.
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
    The objective is linear in the supplied model scores, so the solution puts
    weight only on models with the best mean score. Exactly equal computed
    best means split weight uniformly; near ties are not treated as equal.
    Positive changes of score units preserve the optimum, subject to floating
    precision. No iterative optimizer or SciPy dependency is needed.
    This does not optimize the score of a mixture distribution or the loss
    of an averaged prediction.

    Args:
        score_matrix: Non-empty finite array of shape (n_models, n_test_points). Each
            entry is the score of model i on test point j.
        maximize: If True, maximize the weighted score; if False, minimize.

    Returns:
        Non-negative float64 weights, shape (n_models,), summing to 1.

    Raises:
        ValueError: If scores are empty, non-finite, or not two-dimensional.
        FloatingPointError: If model means cannot be represented numerically.

    Examples:
        >>> weights = stack_scores(np.array([[1.0, 1.0], [3.0, 3.0]]))
        >>> weights.tolist()
        [1.0, 0.0]
        >>> stack_scores(1e6 * np.array([[1.0, 1.0], [3.0, 3.0]])).tolist()
        [1.0, 0.0]
        >>> stack_scores(np.array([[1.0], [1.0], [3.0]])).tolist()
        [0.5, 0.5, 0.0]
    """
    scores = np.asarray(score_matrix, dtype=np.float64)
    if scores.ndim != 2 or 0 in scores.shape or not np.all(np.isfinite(scores)):
        msg = "score_matrix must be a non-empty, finite two-dimensional array"
        raise ValueError(msg)
    # Divide before summing and use the extended accumulator where available.
    # An unrepresentable average must raise instead of producing unchecked weights.
    try:
        with np.errstate(over="raise", invalid="raise"):
            means = np.sum(scores.astype(np.longdouble) / scores.shape[1], axis=1)
    except FloatingPointError as error:
        msg = "score_matrix means are not representable; rescale the score units"
        raise FloatingPointError(msg) from error
    best = np.max(means) if maximize else np.min(means)
    selected = means == best
    return np.asarray(selected / np.count_nonzero(selected), dtype=np.float64)


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
        weights: Stacking weights, shape (n_models,). Normalization leaves the
            supplied array and its backing storage unchanged. Read-only inputs
            and views are supported.

    Returns:
        Weighted average prediction array.

    Examples:
        >>> import numpy as np
        >>> predictions = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
        >>> ensemble_predict(predictions, np.array([0.25, 0.75])).tolist()
        [2.5, 3.5]
        >>> weights = np.array([1.0, 3.0])
        >>> weights.setflags(write=False)
        >>> ensemble_predict(predictions, weights).tolist(), weights.tolist()
        ([2.5, 3.5], [1.0, 3.0])
    """
    w = np.array(weights, dtype=np.float64, copy=True)
    w /= w.sum()
    result = np.zeros_like(predictions[0], dtype=np.float64)
    for pred, wi in zip(predictions, w, strict=True):
        result += wi * pred
    return result
