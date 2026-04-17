"""Study orchestration: hierarchical phases with filtering.

A Study chains Phases, where each phase runs a sweep, scores it,
and optionally filters configs for the next phase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ._pareto import extract_front, hypervolume, pareto_rank
from .protocols import Direction
from .runner import run_adaptive, run_grid
from .stacking import stack_scores

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from .protocols import (
        Annotation,
        Constraint,
        Observable,
        ResultsTable,
        Scorer,
        Simulator,
    )
    from .runner import ProgressCallback

    GridCallable = Callable[[ResultsTable, list[Observable]], list[dict[str, Any]]]


@dataclass
class Phase:
    """A single phase in a multi-phase study.

    Attributes:
        name: Phase identifier (e.g. "discovery", "refinement").
        grid: Explicit config list, ``"carry"`` to re-use filtered configs
            from the previous phase, ``"adaptive"`` for optuna-driven
            search, or a callable ``(ResultsTable, list[Observable]) ->
            list[dict]`` that dynamically generates the grid from the
            previous phase's results.
        filter_fn: Optional callable that takes a ResultsTable and returns
            indices of configs to pass to the next phase. If None, phase
            is terminal.
        n_trials: For adaptive mode, number of optuna trials.
        world: Optional phase-level simulator override.  When set, this
            phase uses *world* instead of the ``Study``-level simulator.
            Useful for multi-fidelity workflows (cheap surrogate first,
            expensive model later).
        scorer: Optional phase-level scorer override.  When set, this
            phase uses *scorer* instead of the ``Study``-level scorer.
    """

    name: str
    grid: list[dict[str, Any]] | str | GridCallable
    filter_fn: Callable[[ResultsTable, list[Observable]], NDArray[np.intp]] | None = (
        None
    )
    n_trials: int = 100
    world: Simulator | None = None
    scorer: Scorer | None = None


def top_k_pareto_filter(
    k: int,
    objective_names: list[str] | None = None,
) -> Callable[[ResultsTable, list[Observable]], NDArray[np.intp]]:
    """Create a filter that keeps the top-K configs by Pareto rank.

    Args:
        k: Maximum number of configs to keep.
        objective_names: Subset of observables to use for ranking.
            If None, uses all observables.

    Returns:
        Filter function compatible with Phase.filter_fn.
    """

    def _filter(
        results: ResultsTable,
        observables: list[Observable],
    ) -> NDArray[np.intp]:
        if objective_names is not None:
            cols = [results.observable_names.index(n) for n in objective_names]
            scores = results.scores[:, cols]
            subset = [o for o in observables if o.name in objective_names]
            dirs = [o.direction for o in subset]
            wts = [o.weight for o in subset]
        else:
            scores = results.scores
            dirs = [o.direction for o in observables]
            wts = [o.weight for o in observables]

        ranks = pareto_rank(scores, dirs, wts)
        order = np.argsort(ranks)
        return order[:k]

    return _filter


def weighted_sum_filter(
    weights: dict[str, float],
    k: int,
) -> Callable[[ResultsTable, list[Observable]], NDArray[np.intp]]:
    """Create a filter that keeps the top-K configs by weighted sum.

    Scalarises multiple objectives into a single score via a weighted sum
    and keeps the ``k`` best configs.  Scores are min-max normalised
    before weighting so that objectives on different scales are
    comparable.  MAXIMIZE objectives are negated before normalisation so
    that lower normalised values are always better.

    Args:
        weights: Mapping from observable name to its scalarisation weight.
            Only the named observables are used; the rest are ignored.
        k: Maximum number of configs to keep.

    Returns:
        Filter function compatible with ``Phase.filter_fn``.
    """

    def _filter(
        results: ResultsTable,
        observables: list[Observable],
    ) -> NDArray[np.intp]:
        obs_lookup = {o.name: o for o in observables}
        cols = [results.observable_names.index(n) for n in weights]
        raw = results.scores[:, cols].copy()

        # Flip MAXIMIZE objectives so lower is always better
        for j, name in enumerate(weights):
            if obs_lookup[name].direction == Direction.MAXIMIZE:
                raw[:, j] = -raw[:, j]

        # Min-max normalise each column to [0, 1]
        col_min = np.nanmin(raw, axis=0)
        col_max = np.nanmax(raw, axis=0)
        span = col_max - col_min
        span[span == 0] = 1.0  # avoid division by zero for constant cols
        normed = (raw - col_min) / span

        w = np.array([weights[n] for n in weights])
        scalar = normed @ w
        order = np.argsort(scalar)
        return order[:k].astype(np.intp)

    return _filter


def feasibility_filter(
    constraints: list[Constraint],
) -> Callable[[ResultsTable, list[Observable]], NDArray[np.intp]]:
    """Create a filter that keeps only designs satisfying all constraints.

    Args:
        constraints: Constraint objects to evaluate against results.

    Returns:
        Filter function compatible with ``Phase.filter_fn``.
    """

    def _filter(
        results: ResultsTable,
        _observables: list[Observable],
    ) -> NDArray[np.intp]:
        mask = results.feasible(constraints)
        return np.nonzero(mask)[0].astype(np.intp)

    return _filter


@dataclass
class Study:
    """Multi-phase model criticism study.

    Attributes:
        world: Simulator generating (truth, observations).
        scorer: Scorer evaluating observables against truth.
        observables: Observable definitions.
        phases: Ordered list of study phases.
        annotations: External information (costs, constraints).
        factors: Factor definitions (needed for adaptive mode).
    """

    world: Simulator
    scorer: Scorer
    observables: list[Observable]
    phases: list[Phase]
    annotations: list[Annotation] = field(default_factory=list)
    factors: list[Any] = field(default_factory=list)

    _results: dict[str, ResultsTable] = field(default_factory=dict, init=False)

    def run(
        self,
        *,
        n_jobs: int = 1,
        callback: ProgressCallback | None = None,
    ) -> None:
        """Execute all phases sequentially.

        Args:
            n_jobs: Number of parallel workers for grid phases.
            callback: Optional progress callback invoked after each trial
                with ``(trial_index, total_trials, trial_result)``.

        Raises:
            ValueError: If a callable grid is used on the first phase
                (no previous results to pass).
        """
        carry_grid: list[dict[str, Any]] | None = None
        prev_result: ResultsTable | None = None

        for phase in self.phases:
            # Resolve phase-level overrides (multi-fidelity support)
            world = phase.world if phase.world is not None else self.world
            scorer = phase.scorer if phase.scorer is not None else self.scorer

            if isinstance(phase.grid, str) and phase.grid == "adaptive":
                result = run_adaptive(
                    world,
                    scorer,
                    self.factors,
                    self.observables,
                    n_trials=phase.n_trials,
                )
            elif callable(phase.grid):
                if prev_result is None:
                    msg = (
                        f"Phase {phase.name!r}: callable grid requires a previous phase"
                    )
                    raise ValueError(msg)
                grid = phase.grid(prev_result, self.observables)
                result = run_grid(
                    world,
                    scorer,
                    grid,
                    self.observables,
                    annotations=self.annotations or None,
                    n_jobs=n_jobs,
                    callback=callback,
                )
            else:
                grid = (
                    phase.grid if isinstance(phase.grid, list) else (carry_grid or [])
                )
                result = run_grid(
                    world,
                    scorer,
                    grid,
                    self.observables,
                    annotations=self.annotations or None,
                    n_jobs=n_jobs,
                    callback=callback,
                )

            self._results[phase.name] = result
            prev_result = result

            if phase.filter_fn is not None:
                keep = phase.filter_fn(result, self.observables)
                carry_grid = [result.configs[i] for i in keep]
            else:
                carry_grid = None

    def results(self, phase: str) -> ResultsTable:
        """Get results for a specific phase.

        Returns:
            ResultsTable for the given phase.
        """
        return self._results[phase]

    def front(self, phase: str) -> NDArray[np.intp]:
        """Get Pareto front indices for a phase.

        Returns:
            Integer array of Pareto-optimal row indices.
        """
        r = self._results[phase]
        dirs = [o.direction for o in self.observables]
        wts = [o.weight for o in self.observables]
        return extract_front(r.scores, dirs, wts)

    def front_hypervolume(
        self,
        phase: str,
        ref_point: NDArray[np.floating[Any]],
    ) -> float:
        """Compute hypervolume of the Pareto front for a phase.

        Returns:
            Hypervolume value.
        """
        r = self._results[phase]
        dirs = [o.direction for o in self.observables]
        wts = [o.weight for o in self.observables]
        front_idx = extract_front(r.scores, dirs, wts)
        return hypervolume(r.scores[front_idx], ref_point, dirs, wts)

    def stack(
        self,
        phase: str,
        *,
        maximize: bool = False,
    ) -> NDArray[np.floating[Any]]:
        """Compute score-based stacking weights for a phase.

        Returns:
            Array of stacking weights.
        """
        r = self._results[phase]
        return stack_scores(r.scores.T, maximize=maximize)

    def summary(self) -> dict[str, dict[str, Any]]:
        """Per-phase summary: n_trials, n_front, observable ranges.

        Returns:
            Dictionary mapping phase names to summary statistics.
        """
        out: dict[str, dict[str, Any]] = {}
        for name, r in self._results.items():
            dirs = [o.direction for o in self.observables]
            wts = [o.weight for o in self.observables]
            front_idx = extract_front(r.scores, dirs, wts)
            out[name] = {
                "n_trials": len(r.configs),
                "n_front": len(front_idx),
                "observable_ranges": {
                    obs: {
                        "min": float(np.nanmin(r.scores[:, i])),
                        "max": float(np.nanmax(r.scores[:, i])),
                    }
                    for i, obs in enumerate(r.observable_names)
                },
            }
        return out
