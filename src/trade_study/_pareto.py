"""Pareto front extraction and performance indicators.

Wraps pymoo for non-dominated sorting and hypervolume computation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .protocols import Direction

if TYPE_CHECKING:
    from numpy.typing import NDArray


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
    from pymoo.util.nds.non_dominated_sorting import (  # type: ignore[import-untyped]
        NonDominatedSorting,
    )

    # pymoo assumes minimization; flip maximize objectives
    obj = scores.copy()
    for j, d in enumerate(directions):
        if d == Direction.MAXIMIZE:
            obj[:, j] = -obj[:, j]

    nds = NonDominatedSorting()
    fronts = nds.do(obj)
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
        NonDominatedSorting,
    )

    obj = scores.copy()
    for j, d in enumerate(directions):
        if d == Direction.MAXIMIZE:
            obj[:, j] = -obj[:, j]

    nds = NonDominatedSorting()
    fronts = nds.do(obj)
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

    obj = front.copy()
    rp = ref_point.copy()
    if directions is not None:
        for j, d in enumerate(directions):
            if d == Direction.MAXIMIZE:
                obj[:, j] = -obj[:, j]
                rp[j] = -rp[j]
    return float(HV(ref_point=rp)(obj))


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

    obj = front.copy()
    ref = reference.copy()
    if directions is not None:
        for j, d in enumerate(directions):
            if d == Direction.MAXIMIZE:
                obj[:, j] = -obj[:, j]
                ref[:, j] = -ref[:, j]
    return float(IGDPlus(ref)(obj))
