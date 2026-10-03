"""Study orchestration: hierarchical phases with filtering.

A Study chains Phases, where each phase runs a sweep, scores it,
and optionally filters configs for the next phase.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ._pareto import extract_front, hypervolume, igd_plus, pareto_rank
from .io import load_results, save_results
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
        n_reps: Number of times to evaluate each design point, forwarded to
            grid and adaptive runners. Adaptive results already contain
            per-trial means; grid results retain raw replicate rows.
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
    n_reps: int = 1
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
        checkpoint_dir: str | Path | None = None,
    ) -> None:
        """Execute all phases sequentially.

        Args:
            n_jobs: Number of parallel workers for grid phases.
            callback: Optional progress callback invoked after each trial
                with ``(trial_index, total_trials, trial_result)``.
            checkpoint_dir: Optional directory for per-phase checkpoints
                (#75). Each completed phase is saved there; on a rerun,
                phases already on disk are loaded instead of run, and
                filters are replayed on them. Results produced elsewhere
                (for example reduced from cluster shards) can be written
                into a phase's directory with :func:`save_results` to
                stand in for running it.

        Raises:
            ValueError: If a callable grid is used on the first phase
                (no previous results to pass).
        """
        carry_grid: list[dict[str, Any]] | None = None
        prev_result: ResultsTable | None = None
        checkpoint = Path(checkpoint_dir) if checkpoint_dir is not None else None
        if checkpoint is not None:
            self._write_index(checkpoint)

        for index, phase in enumerate(self.phases):
            saved = (
                self._phase_dir(checkpoint, index) if checkpoint is not None else None
            )
            if saved is not None and (saved / "meta.json").exists():
                result = load_results(saved)
                self._results[phase.name] = result
                prev_result = result
                carry_grid = self._carry(phase)
                continue

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
                    n_reps=phase.n_reps,
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
                    n_reps=phase.n_reps,
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
                    n_reps=phase.n_reps,
                    callback=callback,
                )

            self._results[phase.name] = result
            prev_result = result
            if saved is not None:
                save_results(result, saved)
            carry_grid = self._carry(phase)

    def _carry(self, phase: Phase) -> list[dict[str, Any]] | None:
        """Return the configs a phase passes on, or ``None`` when terminal.

        Filters run on aggregated per-design-point scores when replicated
        (#112), so filters like ``top_k_pareto_filter`` rank design points
        rather than individual noisy replicates; the raw per-replicate table
        is kept in the study results.
        """
        if phase.filter_fn is None:
            return None
        source = self._design_points(phase)
        keep = phase.filter_fn(source, self.observables)
        return [source.configs[i] for i in keep]

    def _phase_dir(self, root: Path, index: int) -> Path:
        name = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.phases[index].name)
        return root / f"{index:02d}_{name}"

    def _write_index(self, root: Path) -> None:
        """Record the phase order, refusing a checkpoint of another study.

        Raises:
            ValueError: If ``root`` holds a checkpoint with different phases.
        """
        names = [phase.name for phase in self.phases]
        index_path = root / "study.json"
        if index_path.exists():
            recorded = json.loads(index_path.read_text())["phases"]
            if recorded != names:
                msg = (
                    f"Checkpoint at {root} was written for phases {recorded}, "
                    f"not {names}"
                )
                raise ValueError(msg)
            return
        root.mkdir(parents=True, exist_ok=True)
        index_path.write_text(json.dumps({"phases": names}, indent=2))

    def save(self, path: str | Path) -> None:
        """Save every completed phase's results as a checkpoint (#75).

        Args:
            path: Checkpoint directory, compatible with
                ``run(checkpoint_dir=...)``.
        """
        root = Path(path)
        self._write_index(root)
        for index, phase in enumerate(self.phases):
            if phase.name in self._results:
                save_results(self._results[phase.name], self._phase_dir(root, index))

    def load(self, path: str | Path) -> list[str]:
        """Load the completed phases of a checkpoint into this study.

        Args:
            path: Directory written by :meth:`save` or by
                ``run(checkpoint_dir=...)``.

        Returns:
            Names of the phases loaded.

        Raises:
            FileNotFoundError: If ``path`` holds no checkpoint.
        """
        root = Path(path)
        if not (root / "study.json").exists():
            msg = f"No study checkpoint at {root}"
            raise FileNotFoundError(msg)
        self._write_index(root)
        loaded = []
        for index, phase in enumerate(self.phases):
            saved = self._phase_dir(root, index)
            if (saved / "meta.json").exists():
                self._results[phase.name] = load_results(saved)
                loaded.append(phase.name)
        return loaded

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

    def compare_phases(
        self,
        ref_point: NDArray[np.floating[Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Compare the fronts of the completed phases (#81).

        Each phase is summarized over its design points: replicate rows are
        averaged first when the phase ran with ``n_reps > 1``, as for
        filtering, and rows with a non-finite score are left out of the
        front. Each phase after the first is compared with the previous
        completed phase by IGD+ in both directions:

        * ``igd_plus_gain``: IGD+ of the previous front against this
          phase's front, i.e. how far the previous front falls short of this
          one. Positive when this phase found better trade-offs.
        * ``igd_plus_loss``: IGD+ of this phase's front against the previous
          front, i.e. how far this phase falls short of the previous one.
          Zero when this front weakly dominates the previous front.

        Args:
            ref_point: Hypervolume reference point, in the observables'
                units. Defaults to the worst value of each observable over
                every phase's front, pushed outward by 10% of its range (by 1
                when the range is zero), so hypervolumes are comparable
                across phases.

        Returns:
            One dict per completed phase, in phase order, with ``phase``,
            ``n_trials`` (design points), ``n_front``, ``hypervolume``,
            ``best`` (best value of each observable over the phase's design
            points) and ``igd_plus_gain``/``igd_plus_loss`` (``None`` for the
            first phase). Phases without a finite front report NaN.
        """
        dirs = [o.direction for o in self.observables]
        wts = [o.weight for o in self.observables]
        tables = [
            (phase.name, self._design_points(phase))
            for phase in self.phases
            if phase.name in self._results
        ]
        fronts = {
            name: _finite_front(table.scores, dirs, wts) for name, table in tables
        }
        reference = (
            np.asarray(ref_point, dtype=float)
            if ref_point is not None
            else _padded_worst(list(fronts.values()), dirs)
        )
        rows: list[dict[str, Any]] = []
        previous: NDArray[np.floating[Any]] | None = None
        for name, table in tables:
            front = fronts[name]
            row: dict[str, Any] = {
                "phase": name,
                "n_trials": len(table.configs),
                "n_front": len(front),
                "hypervolume": (
                    hypervolume(front, reference, dirs, wts) if len(front) else np.nan
                ),
                "best": _best_values(table, self.observables),
                "igd_plus_gain": None,
                "igd_plus_loss": None,
            }
            if previous is not None:
                comparable = len(front) > 0 and len(previous) > 0
                row["igd_plus_gain"] = (
                    igd_plus(previous, front, dirs, wts) if comparable else np.nan
                )
                row["igd_plus_loss"] = (
                    igd_plus(front, previous, dirs, wts) if comparable else np.nan
                )
            rows.append(row)
            previous = front
        return rows

    def _design_points(self, phase: Phase) -> ResultsTable:
        result = self._results[phase.name]
        if phase.n_reps > 1 and phase.grid != "adaptive":
            return result.aggregate_replicates()
        return result

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


def _finite_front(
    scores: NDArray[np.floating[Any]],
    directions: list[Direction],
    weights: list[float],
) -> NDArray[np.floating[Any]]:
    """Return the Pareto-optimal rows among rows with finite scores.

    Returns:
        Front scores, shape ``(n_front, n_observables)``.
    """
    finite = scores[np.all(np.isfinite(scores), axis=1)]
    if len(finite) == 0:
        return finite
    return finite[extract_front(finite, directions, weights)]


def _padded_worst(
    fronts: list[NDArray[np.floating[Any]]],
    directions: list[Direction],
) -> NDArray[np.floating[Any]]:
    """Return the worst front value per observable, pushed outward.

    Returns:
        Reference point dominated by every front point.
    """
    points = [front for front in fronts if len(front)]
    if not points:
        return np.zeros(len(directions))
    union = np.vstack(points)
    low, high = union.min(axis=0), union.max(axis=0)
    pad = np.where(high > low, 0.1 * (high - low), 1.0)
    maximize = np.array([d == Direction.MAXIMIZE for d in directions])
    worst: NDArray[np.floating[Any]] = np.where(maximize, low - pad, high + pad)
    return worst


def _best_values(
    table: ResultsTable,
    observables: list[Observable],
) -> dict[str, float]:
    """Return the best finite value of each observable.

    Returns:
        Mapping from observable name to its best value (NaN if none).
    """
    best: dict[str, float] = {}
    for obs in observables:
        column = table.scores[:, table.observable_names.index(obs.name)]
        column = column[np.isfinite(column)]
        if len(column) == 0:
            best[obs.name] = float("nan")
        elif obs.direction == Direction.MAXIMIZE:
            best[obs.name] = float(column.max())
        else:
            best[obs.name] = float(column.min())
    return best
