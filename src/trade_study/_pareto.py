"""Pareto front extraction and performance indicators.

Wraps pymoo for non-dominated sorting and hypervolume computation. Every public
function here needs the optional ``pareto`` extra (``pip install
'trade-study[pareto]'``) and raises an ``ImportError`` that says so when pymoo is
not installed.
"""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING, Any

import numpy as np

from .protocols import Direction

if TYPE_CHECKING:
    from numpy.typing import NDArray

_PYMOO_MISSING = (
    "pymoo is required for Pareto operations but is not installed; "
    "install it with `pip install 'trade-study[pareto]'`."
)


def _require_pymoo() -> None:
    """Fail early and clearly when the optional ``pareto`` extra is absent.

    Raises:
        ImportError: If ``pymoo`` cannot be imported, naming the extra to install.
    """
    if importlib.util.find_spec("pymoo") is None:
        raise ImportError(_PYMOO_MISSING)


def _normalize_objectives(
    scores: NDArray[np.floating[Any]],
    directions: list[Direction],
    weights: list[float] | None = None,
) -> NDArray[np.floating[Any]]:
    """Convert to minimisation space and apply optional weights.

    Args:
        scores: Array of shape ``(n, m)``.
        directions: Optimization direction per objective.
        weights: If provided, each column is scaled by its weight before
            non-dominated sorting.

    Returns:
        Transformed score array (copy, never mutates input).
    """
    obj = scores.copy()
    for j, d in enumerate(directions):
        if d == Direction.MAXIMIZE:
            obj[:, j] = -obj[:, j]
    if weights is not None:
        for j, w in enumerate(weights):
            obj[:, j] *= w
    return obj


def extract_front(
    scores: NDArray[np.floating[Any]],
    directions: list[Direction],
    weights: list[float] | None = None,
) -> NDArray[np.intp]:
    """Extract Pareto-optimal indices from a score matrix.

    Args:
        scores: Array of shape (n_trials, n_objectives).
        directions: Optimization direction for each objective.
        weights: Optional objective scaling. Strictly positive weights preserve
            dominance and front membership; they do not encode preferences.

    Returns:
        Integer array of row indices on the Pareto front.

    Examples:
        >>> scores = np.array([[1.0, 1.0], [2.0, 2.0], [0.5, 3.0]])
        >>> directions = [Direction.MINIMIZE, Direction.MINIMIZE]
        >>> sorted(extract_front(scores, directions).tolist())
        [0, 2]
    """
    _require_pymoo()
    from pymoo.util.nds.non_dominated_sorting import (  # type: ignore[import-untyped]
        NonDominatedSorting,
    )

    obj = _normalize_objectives(scores, directions, weights)
    nds = NonDominatedSorting()
    fronts = nds.do(obj)
    return np.asarray(fronts[0], dtype=np.intp)


def pareto_rank(
    scores: NDArray[np.floating[Any]],
    directions: list[Direction],
    weights: list[float] | None = None,
) -> NDArray[np.intp]:
    """Assign Pareto rank to each trial (0 = front, 1 = next layer, ...).

    Args:
        scores: Array of shape (n_trials, n_objectives).
        directions: Optimization direction for each objective.
        weights: Optional per-objective weights.

    Returns:
        Integer array of ranks, shape (n_trials,).

    Examples:
        >>> scores = np.array([[1.0, 1.0], [2.0, 2.0]])
        >>> pareto_rank(scores, [Direction.MINIMIZE] * 2).tolist()
        [0, 1]
    """
    _require_pymoo()
    from pymoo.util.nds.non_dominated_sorting import (
        NonDominatedSorting,
    )

    obj = _normalize_objectives(scores, directions, weights)
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
    weights: list[float] | None = None,
) -> float:
    """Compute hypervolume indicator for a Pareto front.

    Args:
        front: Array of shape (n_points, n_objectives) on the front.
        ref_point: Reference point worse than all front points in every
            objective after direction normalization.
        directions: If provided, flips maximize objectives before computing.
        weights: Optional per-objective weights applied after direction flip.

    Returns:
        Hypervolume value.
    """
    _require_pymoo()
    from pymoo.indicators.hv import HV  # type: ignore[import-untyped]

    obj = front.copy()
    rp = ref_point.copy()
    if directions is not None:
        for j, d in enumerate(directions):
            if d == Direction.MAXIMIZE:
                obj[:, j] = -obj[:, j]
                rp[j] = -rp[j]
    if weights is not None:
        for j, w in enumerate(weights):
            obj[:, j] *= w
            rp[j] *= w
    return float(HV(ref_point=rp)(obj))


def igd_plus(
    front: NDArray[np.floating[Any]],
    reference: NDArray[np.floating[Any]],
    directions: list[Direction] | None = None,
    weights: list[float] | None = None,
) -> float:
    """Compute IGD+ indicator.

    Args:
        front: Obtained Pareto front.
        reference: Reference Pareto front.
        directions: Optimization directions.
        weights: Optional per-objective weights applied after direction flip.

    Returns:
        IGD+ value (lower is better).
    """
    _require_pymoo()
    from pymoo.indicators.igd_plus import IGDPlus  # type: ignore[import-untyped]

    obj = front.copy()
    ref = reference.copy()
    if directions is not None:
        for j, d in enumerate(directions):
            if d == Direction.MAXIMIZE:
                obj[:, j] = -obj[:, j]
                ref[:, j] = -ref[:, j]
    if weights is not None:
        for j, w in enumerate(weights):
            obj[:, j] *= w
            ref[:, j] *= w
    return float(IGDPlus(ref)(obj))
