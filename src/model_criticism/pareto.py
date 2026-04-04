"""Pareto front extraction and performance indicators.

Wraps pymoo for non-dominated sorting and hypervolume computation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .protocols import Direction


def extract_front(
    scores: NDArray[np.floating[Any]],
    directions: list[Direction],
) -> NDArray[np.intp]:
    """Extract Pareto-optimal indices from a score matrix.

    Args:
        scores: Array of shape (n_trials, n_objectives).
        directions: Optimization direction for each objective.

    Returns:
        Integer array of row indices on the Pareto front.
    """
    from pymoo.util.nds.non_dominated_sorting import (
        NonDominatedSorting,  # type: ignore[import-untyped]
    )

    # pymoo assumes minimization; flip maximize objectives
    F = scores.copy()
    for j, d in enumerate(directions):
        if d == Direction.MAXIMIZE:
            F[:, j] = -F[:, j]

    nds = NonDominatedSorting()
    fronts = nds.do(F)
    return np.asarray(fronts[0], dtype=np.intp)


def pareto_rank(
    scores: NDArray[np.floating[Any]],
    directions: list[Direction],
) -> NDArray[np.intp]:
    """Assign Pareto rank to each trial (0 = front, 1 = next layer, ...).

    Args:
        scores: Array of shape (n_trials, n_objectives).
        directions: Optimization direction for each objective.

    Returns:
        Integer array of ranks, shape (n_trials,).
    """
    from pymoo.util.nds.non_dominated_sorting import (
        NonDominatedSorting,  # type: ignore[import-untyped]
    )

    F = scores.copy()
    for j, d in enumerate(directions):
        if d == Direction.MAXIMIZE:
            F[:, j] = -F[:, j]

    nds = NonDominatedSorting()
    fronts = nds.do(F)
    ranks = np.empty(len(scores), dtype=np.intp)
    for rank, front in enumerate(fronts):
        ranks[front] = rank
    return ranks


def hypervolume(
    front: NDArray[np.floating[Any]],
    ref_point: NDArray[np.floating[Any]],
    directions: list[Direction] | None = None,
) -> float:
    """Compute hypervolume indicator for a Pareto front.

    Args:
        front: Array of shape (n_points, n_objectives) on the front.
        ref_point: Reference point (should dominate all front points after
            direction normalization).
        directions: If provided, flips maximize objectives before computing.

    Returns:
        Hypervolume value.
    """
    from pymoo.indicators.hv import HV  # type: ignore[import-untyped]

    F = front.copy()
    rp = ref_point.copy()
    if directions is not None:
        for j, d in enumerate(directions):
            if d == Direction.MAXIMIZE:
                F[:, j] = -F[:, j]
                rp[j] = -rp[j]
    return float(HV(ref_point=rp)(F))


def igd_plus(
    front: NDArray[np.floating[Any]],
    reference: NDArray[np.floating[Any]],
    directions: list[Direction] | None = None,
) -> float:
    """Compute IGD+ indicator.

    Args:
        front: Obtained Pareto front.
        reference: Reference Pareto front.
        directions: Optimization directions.

    Returns:
        IGD+ value (lower is better).
    """
    from pymoo.indicators.igd_plus import IGDPlus  # type: ignore[import-untyped]

    F = front.copy()
    R = reference.copy()
    if directions is not None:
        for j, d in enumerate(directions):
            if d == Direction.MAXIMIZE:
                F[:, j] = -F[:, j]
                R[:, j] = -R[:, j]
    return float(IGDPlus(R)(F))
