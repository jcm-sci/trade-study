"""Study orchestration: hierarchical phases with filtering.

A Study chains Phases, where each phase runs a sweep, scores it,
and optionally filters configs for the next phase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ._pareto import extract_front, hypervolume, pareto_rank
from .runner import run_adaptive, run_grid
from .stacking import stack_scores

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from .protocols import (
        Annotation,
        Observable,
        ResultsTable,
        Scorer,
        Simulator,
    )

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
    """

    name: str
    grid: list[dict[str, Any]] | str | GridCallable
    filter_fn: Callable[[ResultsTable, list[Observable]], NDArray[np.intp]] | None = (
        None
    )
    n_trials: int = 100


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
            dirs = [o.direction for o in observables if o.name in objective_names]
        else:
            scores = results.scores
            dirs = [o.direction for o in observables]

        ranks = pareto_rank(scores, dirs)
        order = np.argsort(ranks)
        return order[:k]

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

    def run(self, *, n_jobs: int = 1) -> None:
        """Execute all phases sequentially.

        Raises:
            ValueError: If a callable grid is used on the first phase
                (no previous results to pass).
        """
        carry_grid: list[dict[str, Any]] | None = None
        prev_result: ResultsTable | None = None

        for phase in self.phases:
            if isinstance(phase.grid, str) and phase.grid == "adaptive":
                result = run_adaptive(
                    self.world,
                    self.scorer,
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
                    self.world,
                    self.scorer,
                    grid,
                    self.observables,
                    annotations=self.annotations or None,
                    n_jobs=n_jobs,
                )
            else:
                grid = (
                    phase.grid if isinstance(phase.grid, list) else (carry_grid or [])
                )
                result = run_grid(
                    self.world,
                    self.scorer,
                    grid,
                    self.observables,
                    annotations=self.annotations or None,
                    n_jobs=n_jobs,
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
        return extract_front(r.scores, dirs)

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
        front_idx = extract_front(r.scores, dirs)
        return hypervolume(r.scores[front_idx], ref_point, dirs)

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
            front_idx = extract_front(r.scores, dirs)
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
