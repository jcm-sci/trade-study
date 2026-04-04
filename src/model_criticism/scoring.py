"""Scoring functions wrapping scoringrules and scipy.

Provides a uniform ``score(metric, predictions, truth)`` interface
for all proper scoring rules and calibration diagnostics.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def score(
    metric: str,
    predictions: NDArray[np.floating[Any]],
    truth: NDArray[np.floating[Any]],
    **kwargs: Any,
) -> float:
    """Compute a scalar scoring rule.

    Args:
        metric: One of "crps", "wis", "interval", "energy", "log_score",
            "rmse", "mae", "coverage", "brier".
        predictions: Model predictions (ensemble members, quantiles, etc.).
        truth: Known ground truth values.
        **kwargs: Metric-specific arguments (e.g. ``alpha`` for interval score,
            ``level`` for coverage).

    Returns:
        Scalar score value.
    """
    if metric == "crps":
        return _crps(predictions, truth)
    if metric == "wis":
        return _wis(predictions, truth, **kwargs)
    if metric == "interval":
        return _interval(predictions, truth, **kwargs)
    if metric == "rmse":
        return float(np.sqrt(np.mean((predictions - truth) ** 2)))
    if metric == "mae":
        return float(np.mean(np.abs(predictions - truth)))
    if metric == "coverage":
        return _coverage(predictions, truth, **kwargs)
    if metric == "energy":
        return _energy(predictions, truth)
    if metric == "brier":
        return _brier(predictions, truth)
    msg = f"Unknown metric: {metric!r}"
    raise ValueError(msg)


def _crps(
    ensemble: NDArray[np.floating[Any]],
    truth: NDArray[np.floating[Any]],
) -> float:
    """CRPS via scoringrules."""
    import scoringrules as sr  # type: ignore[import-untyped]

    return float(np.mean(sr.crps_ensemble(truth, ensemble)))


def _wis(
    predictions: NDArray[np.floating[Any]],
    truth: NDArray[np.floating[Any]],
    **kwargs: Any,
) -> float:
    """Weighted interval score via scoringrules."""
    import scoringrules as sr  # type: ignore[import-untyped]

    alpha = kwargs.get("alpha", np.array([0.02, 0.05, 0.1, 0.2, 0.5]))
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
    **kwargs: Any,
) -> float:
    """Interval score via scoringrules."""
    import scoringrules as sr  # type: ignore[import-untyped]

    alpha = kwargs.get("alpha", 0.05)
    return float(
        np.mean(
            sr.interval_score(truth, predictions[..., 0], predictions[..., 1], alpha),
        ),
    )


def _coverage(
    predictions: NDArray[np.floating[Any]],
    truth: NDArray[np.floating[Any]],
    **kwargs: Any,
) -> float:
    """Empirical coverage rate at a given nominal level."""
    level = kwargs.get("level", 0.95)
    alpha = 1.0 - level
    lower = np.quantile(predictions, alpha / 2, axis=-1)
    upper = np.quantile(predictions, 1 - alpha / 2, axis=-1)
    return float(np.mean((truth >= lower) & (truth <= upper)))


def _energy(
    ensemble: NDArray[np.floating[Any]],
    truth: NDArray[np.floating[Any]],
) -> float:
    """Energy score via scoringrules."""
    import scoringrules as sr  # type: ignore[import-untyped]

    return float(np.mean(sr.energy_score(truth, ensemble)))


def _brier(
    predictions: NDArray[np.floating[Any]],
    truth: NDArray[np.floating[Any]],
) -> float:
    """Brier score via scoringrules."""
    import scoringrules as sr  # type: ignore[import-untyped]

    return float(np.mean(sr.brier_score(truth, predictions)))


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
    empirical = np.array(
        [_coverage(posteriors, truth, level=float(lv)) for lv in levels]
    )
    return levels, empirical
