"""Scoring functions wrapping scoringrules and scipy.

Provides a uniform ``score(metric, predictions, truth)`` interface
for all proper scoring rules and calibration diagnostics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def score(
    metric: str,
    predictions: NDArray[np.floating[Any]],
    truth: NDArray[np.floating[Any]],
    *,
    alpha: float | NDArray[np.floating[Any]] | None = None,
    level: float = 0.95,
) -> float:
    """Compute a scalar scoring rule.

    Args:
        metric: One of "crps", "wis", "interval", "energy",
            "rmse", "mae", "coverage", "brier".
        predictions: Model predictions (ensemble members, quantiles, etc.).
        truth: Known ground truth values.
        alpha: Significance level for interval-based scores.
        level: Nominal coverage level for coverage metric.

    Returns:
        Scalar score value.

    Raises:
        ValueError: If the metric name is not recognized.
    """
    simple = {
        "crps": _crps,
        "energy": _energy,
        "brier": _brier,
        "rmse": _rmse,
        "mae": _mae,
    }
    if metric in simple:
        return simple[metric](predictions, truth)
    if metric == "wis":
        return _wis(predictions, truth, alpha=alpha)
    if metric == "interval":
        return _interval(predictions, truth, alpha=alpha)
    if metric == "coverage":
        return _coverage(predictions, truth, level=level)
    msg = f"Unknown metric: {metric!r}"
    raise ValueError(msg)


def _crps(
    ensemble: NDArray[np.floating[Any]],
    truth: NDArray[np.floating[Any]],
) -> float:
    """CRPS via scoringrules.

    Returns:
        Mean CRPS across observations.
    """
    import scoringrules as sr  # type: ignore[import-untyped]

    return float(np.mean(sr.crps_ensemble(truth, ensemble)))


def _wis(
    predictions: NDArray[np.floating[Any]],
    truth: NDArray[np.floating[Any]],
    *,
    alpha: float | NDArray[np.floating[Any]] | None = None,
) -> float:
    """Weighted interval score via scoringrules.

    Returns:
        Mean WIS across observations.
    """
    import scoringrules as sr

    if alpha is None:
        alpha = np.array([0.02, 0.05, 0.1, 0.2, 0.5])
    return float(
        np.mean(
            sr.weighted_interval_score(
                truth,
                predictions[..., 0],
                predictions[..., 1],
                alpha,
            ),
        ),
    )


def _interval(
    predictions: NDArray[np.floating[Any]],
    truth: NDArray[np.floating[Any]],
    *,
    alpha: float | NDArray[np.floating[Any]] | None = None,
) -> float:
    """Interval score via scoringrules.

    Returns:
        Mean interval score across observations.
    """
    import scoringrules as sr

    if alpha is None:
        alpha = 0.05
    return float(
        np.mean(
            sr.interval_score(truth, predictions[..., 0], predictions[..., 1], alpha),
        ),
    )


def _coverage(
    predictions: NDArray[np.floating[Any]],
    truth: NDArray[np.floating[Any]],
    *,
    level: float = 0.95,
) -> float:
    """Empirical coverage rate at a given nominal level.

    Returns:
        Fraction of truth values within the predicted interval.
    """
    cov_alpha = 1.0 - level
    lower = np.quantile(predictions, cov_alpha / 2, axis=-1)
    upper = np.quantile(predictions, 1 - cov_alpha / 2, axis=-1)
    return float(np.mean((truth >= lower) & (truth <= upper)))


def _energy(
    ensemble: NDArray[np.floating[Any]],
    truth: NDArray[np.floating[Any]],
) -> float:
    """Energy score via scoringrules.

    Returns:
        Mean energy score across observations.
    """
    import scoringrules as sr

    return float(np.mean(sr.energy_score(truth, ensemble)))


def _brier(
    predictions: NDArray[np.floating[Any]],
    truth: NDArray[np.floating[Any]],
) -> float:
    """Brier score via scoringrules.

    Returns:
        Mean Brier score across observations.
    """
    import scoringrules as sr

    return float(np.mean(sr.brier_score(truth, predictions)))


def _rmse(
    predictions: NDArray[np.floating[Any]],
    truth: NDArray[np.floating[Any]],
) -> float:
    """Root mean squared error.

    Returns:
        RMSE value.
    """
    return float(np.sqrt(np.mean((predictions - truth) ** 2)))


def _mae(
    predictions: NDArray[np.floating[Any]],
    truth: NDArray[np.floating[Any]],
) -> float:
    """Mean absolute error.

    Returns:
        MAE value.
    """
    return float(np.mean(np.abs(predictions - truth)))


def coverage_curve(
    posteriors: NDArray[np.floating[Any]],
    truth: NDArray[np.floating[Any]],
    levels: NDArray[np.floating[Any]] | None = None,
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    """Compute empirical coverage across nominal levels.

    Args:
        posteriors: Posterior samples, shape (n_obs, n_samples).
        truth: True values, shape (n_obs,).
        levels: Nominal coverage levels (default: 0.05 to 0.99).

    Returns:
        Tuple of (nominal_levels, empirical_coverage).
    """
    if levels is None:
        levels = np.linspace(0.05, 0.99, 50)
    empirical = np.array([
        _coverage(posteriors, truth, level=float(lv)) for lv in levels
    ])
    return levels, empirical
