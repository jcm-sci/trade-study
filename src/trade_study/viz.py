"""Visualization utilities for trade-study results.

All functions require ``matplotlib`` (optional dependency, install via
``pip install trade-study[viz]``).  Each function returns a
``(Figure, Axes)`` tuple for composability with user layouts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .protocols import Direction

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.colors import Colormap
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

    from .protocols import ResultsTable


def _require_matplotlib() -> None:
    """Raise a helpful error if matplotlib is not installed.

    Raises:
        ImportError: If matplotlib is not available.
    """
    try:
        import matplotlib as mpl  # noqa: F401
    except ImportError as exc:
        msg = (
            "matplotlib is required for visualization. "
            "Install it with:  pip install trade-study[viz]"
        )
        raise ImportError(msg) from exc


def plot_front(
    results: ResultsTable,
    directions: list[Direction],
    *,
    ax: Axes | None = None,
    front_kw: dict[str, Any] | None = None,
    dominated_kw: dict[str, Any] | None = None,
) -> tuple[Figure, Axes | np.ndarray[Any, np.dtype[Any]]]:
    """Plot a Pareto front from a results table.

    For two objectives, draws a 2-D scatter.  For three objectives,
    draws a 3-D scatter.  For four or more, draws a pairwise scatter
    matrix of the first three objectives.

    Args:
        results: Scored results from a study phase.
        directions: Optimization direction per observable.
        ax: Optional axes to draw on (only used for 2-D case).
        front_kw: Extra keyword arguments for front-point scatter.
        dominated_kw: Extra keyword arguments for dominated-point scatter.

    Returns:
        Tuple of (Figure, Axes).  For the pairwise matrix case the
        second element is an ndarray of Axes.

    Raises:
        ValueError: If fewer than two objectives are present.
    """
    _require_matplotlib()

    from ._pareto import extract_front

    n_obj = results.scores.shape[1]
    if n_obj < 2:
        msg = "plot_front requires at least 2 objectives"
        raise ValueError(msg)

    front_idx = extract_front(results.scores, directions)
    is_front = np.zeros(len(results.scores), dtype=bool)
    is_front[front_idx] = True

    fkw: dict[str, Any] = {
        "s": 40,
        "zorder": 3,
        "label": "Pareto front",
    }
    if front_kw:
        fkw.update(front_kw)

    dkw: dict[str, Any] = {
        "s": 20,
        "alpha": 0.35,
        "color": "0.6",
        "zorder": 2,
        "label": "Dominated",
    }
    if dominated_kw:
        dkw.update(dominated_kw)

    names = results.observable_names

    if n_obj == 2:
        return _plot_front_2d(results.scores, is_front, names, ax, fkw, dkw)
    if n_obj == 3:
        return _plot_front_3d(results.scores, is_front, names, fkw, dkw)
    return _plot_front_pairs(results.scores, is_front, names, fkw, dkw)


def _plot_front_2d(
    scores: NDArray[np.floating[Any]],
    is_front: NDArray[np.bool_],
    names: list[str],
    ax: Axes | None,
    fkw: dict[str, Any],
    dkw: dict[str, Any],
) -> tuple[Figure, Axes]:
    """Draw a 2-D Pareto front scatter.

    Args:
        scores: Score matrix of shape ``(n_trials, 2)``.
        is_front: Boolean mask of Pareto-front membership.
        names: Observable names for axis labels.
        ax: Optional pre-existing axes.
        fkw: Scatter keyword arguments for front points.
        dkw: Scatter keyword arguments for dominated points.

    Returns:
        Tuple of (Figure, Axes).
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()  # type: ignore[assignment]

    ax.scatter(scores[~is_front, 0], scores[~is_front, 1], **dkw)
    ax.scatter(scores[is_front, 0], scores[is_front, 1], **fkw)
    ax.set_xlabel(names[0])
    ax.set_ylabel(names[1])
    ax.legend()
    return fig, ax


def _plot_front_3d(
    scores: NDArray[np.floating[Any]],
    is_front: NDArray[np.bool_],
    names: list[str],
    fkw: dict[str, Any],
    dkw: dict[str, Any],
) -> tuple[Figure, Axes]:
    """Draw a 3-D Pareto front scatter.

    Args:
        scores: Score matrix of shape ``(n_trials, 3)``.
        is_front: Boolean mask of Pareto-front membership.
        names: Observable names for axis labels.
        fkw: Scatter keyword arguments for front points.
        dkw: Scatter keyword arguments for dominated points.

    Returns:
        Tuple of (Figure, Axes).
    """
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        scores[~is_front, 0],
        scores[~is_front, 1],
        scores[~is_front, 2],
        **dkw,
    )
    ax.scatter(
        scores[is_front, 0],
        scores[is_front, 1],
        scores[is_front, 2],
        **fkw,
    )
    ax.set_xlabel(names[0])
    ax.set_ylabel(names[1])
    ax.set_zlabel(names[2])
    ax.legend()
    return fig, ax


def _plot_front_pairs(
    scores: NDArray[np.floating[Any]],
    is_front: NDArray[np.bool_],
    names: list[str],
    fkw: dict[str, Any],
    dkw: dict[str, Any],
) -> tuple[Figure, np.ndarray[Any, np.dtype[Any]]]:
    """Draw pairwise scatter matrix for the first 3 objectives.

    Args:
        scores: Score matrix of shape ``(n_trials, n_obj)``.
        is_front: Boolean mask of Pareto-front membership.
        names: Observable names for axis labels.
        fkw: Scatter keyword arguments for front points.
        dkw: Scatter keyword arguments for dominated points.

    Returns:
        Tuple of (Figure, ndarray of Axes).
    """
    import matplotlib.pyplot as plt

    idx = list(range(min(len(names), 3)))
    n = len(idx)
    fig, axes = plt.subplots(n, n, figsize=(4 * n, 4 * n))

    for row in range(n):
        for col in range(n):
            ax = axes[row, col]
            if row == col:
                ax.hist(scores[:, idx[row]], bins=20, color="0.7")
                ax.set_xlabel(names[idx[row]])
            else:
                ax.scatter(
                    scores[~is_front, idx[col]],
                    scores[~is_front, idx[row]],
                    **dkw,
                )
                ax.scatter(
                    scores[is_front, idx[col]],
                    scores[is_front, idx[row]],
                    **fkw,
                )
                if col == 0:
                    ax.set_ylabel(names[idx[row]])
                if row == n - 1:
                    ax.set_xlabel(names[idx[col]])

    fig.tight_layout()
    return fig, axes


def _normalize_parallel(
    scores: NDArray[np.floating[Any]],
    directions: list[Direction],
) -> NDArray[np.floating[Any]]:
    """Normalize scores to [0, 1] with 'better' mapped to 1.

    Args:
        scores: Raw score matrix of shape ``(n_trials, n_obj)``.
        directions: Optimization direction per objective.

    Returns:
        Normalized array, same shape as *scores*.
    """
    lo = scores.min(axis=0)
    hi = scores.max(axis=0)
    span = np.where(hi - lo > 0, hi - lo, 1.0)
    normed: NDArray[np.floating[Any]] = np.asarray((scores - lo) / span)
    for j, d in enumerate(directions):
        if d == Direction.MINIMIZE:
            normed[:, j] = 1.0 - normed[:, j]
    return normed


def _build_parallel_lines(
    normed: NDArray[np.floating[Any]],
    ranks: NDArray[np.intp],
    cm: Colormap,
) -> tuple[list[NDArray[np.floating[Any]]], list[tuple[float, ...]]]:
    """Build ordered line segments and colors for parallel coordinates.

    Args:
        normed: Normalized score matrix of shape ``(n_trials, n_obj)``.
        ranks: Pareto rank per trial.
        cm: Matplotlib colormap instance.

    Returns:
        Tuple of (segments, colors), both ordered dominated-first.
    """
    n_trials, n_obj = normed.shape
    max_rank = max(int(ranks.max()), 1)
    rank_norm = ranks / max_rank
    x = np.arange(n_obj, dtype=np.float64)

    segments = [np.column_stack([x, normed[i]]) for i in range(n_trials)]
    colors = [cm(1.0 - rank_norm[i]) for i in range(n_trials)]

    order = np.argsort(-ranks)
    return (
        [segments[i] for i in order],
        [colors[i] for i in order],
    )


def plot_parallel(
    results: ResultsTable,
    directions: list[Direction],
    *,
    ax: Axes | None = None,
    cmap: str = "viridis",
) -> tuple[Figure, Axes]:
    """Parallel coordinates plot colored by Pareto rank.

    Each vertical axis represents one observable, normalized to [0, 1]
    with the "better" end pointing up.  Lines are colored by Pareto
    rank (0 = front, darker = better).

    Args:
        results: Scored results from a study phase.
        directions: Optimization direction per observable.
        ax: Optional axes to draw on.
        cmap: Matplotlib colormap name for Pareto-rank coloring.

    Returns:
        Tuple of (Figure, Axes).
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize as MplNormalize

    from ._pareto import pareto_rank

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(6, len(directions) * 1.5), 5))
    else:
        fig = ax.get_figure()  # type: ignore[assignment]

    scores = results.scores
    n_obj = scores.shape[1]
    normed = _normalize_parallel(scores, directions)
    ranks = pareto_rank(scores, directions)
    max_rank = max(int(ranks.max()), 1)

    cm = plt.get_cmap(cmap)
    segments, colors = _build_parallel_lines(normed, ranks, cm)

    lc = LineCollection(segments, colors=colors, linewidths=1.2, alpha=0.7)
    ax.add_collection(lc)

    ax.set_xlim(-0.1, n_obj - 0.9)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(np.arange(n_obj))
    ax.set_xticklabels(results.observable_names)
    ax.set_ylabel("Normalized score (↑ better)")

    sm = plt.cm.ScalarMappable(
        cmap=cm,
        norm=MplNormalize(0, max_rank),
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, label="Pareto rank")
    cbar.ax.invert_yaxis()

    return fig, ax


def plot_calibration(
    nominal: NDArray[np.floating[Any]],
    empirical: NDArray[np.floating[Any]],
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot a calibration curve from ``coverage_curve()`` output.

    Compares nominal coverage levels against empirical coverage.
    A well-calibrated model follows the diagonal.

    Args:
        nominal: Nominal coverage levels, shape (n_levels,).
        empirical: Empirical coverage values, shape (n_levels,).
        ax: Optional axes to draw on.

    Returns:
        Tuple of (Figure, Axes).
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()  # type: ignore[assignment]

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Ideal")
    ax.plot(nominal, empirical, "o-", markersize=3, label="Empirical")
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Empirical coverage")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend()

    return fig, ax


def plot_scores(
    results: ResultsTable,
    observable: str,
    directions: list[Direction] | None = None,
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Strip plot of one observable across all configurations.

    Each dot is one trial.  If *directions* are provided, Pareto-front
    designs are highlighted.

    Args:
        results: Scored results from a study phase.
        observable: Name of the observable to plot.
        directions: If given, highlight Pareto-front designs.
        ax: Optional axes to draw on.

    Returns:
        Tuple of (Figure, Axes).

    Raises:
        ValueError: If the observable name is not found.
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt

    if observable not in results.observable_names:
        msg = f"Observable {observable!r} not in {results.observable_names}"
        raise ValueError(msg)

    col = results.observable_names.index(observable)
    values = results.scores[:, col]

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()  # type: ignore[assignment]

    jitter = np.random.default_rng(0).uniform(-0.15, 0.15, size=len(values))

    if directions is not None:
        from ._pareto import extract_front

        front_idx = extract_front(results.scores, directions)
        is_front = np.zeros(len(values), dtype=bool)
        is_front[front_idx] = True

        ax.scatter(
            jitter[~is_front],
            values[~is_front],
            s=20,
            alpha=0.4,
            color="0.6",
            label="Dominated",
        )
        ax.scatter(
            jitter[is_front],
            values[is_front],
            s=40,
            zorder=3,
            label="Pareto front",
        )
        ax.legend()
    else:
        ax.scatter(jitter, values, s=20, alpha=0.6)

    ax.set_ylabel(observable)
    ax.set_xticks([])

    return fig, ax
