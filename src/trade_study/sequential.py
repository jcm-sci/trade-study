"""Budgeted replication with simultaneous finite-horizon confidence bounds."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from .protocols import Direction, ResultsTable
from .runner import _generate_accepts_rep, _run_single

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .protocols import Constraint, Observable, Scorer, Simulator, TrialResult


@dataclass(frozen=True)
class ReplicationPolicy:
    """Explicit budget and bounded-score assumptions for sequential replication.

    Attributes:
        score_bounds: Known almost-sure bounds in raw observable units.
        min_reps: Initial evaluations per design.
        max_reps: Maximum evaluations per design and inference horizon.
        max_evaluations: Total simulator/scorer evaluations allowed.
        confidence: Joint coverage across all designs, observables and looks.
        equivalence: Practical-equivalence tolerances in raw observable units;
            omitted observables use zero.
    """

    score_bounds: dict[str, tuple[float, float]]
    min_reps: int = 2
    max_reps: int = 100
    max_evaluations: int = 1000
    confidence: float = 0.95
    equivalence: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class SequentialResult:
    """Raw replicates, allocation decisions, and simultaneous mean intervals."""

    results: ResultsTable
    summary: ResultsTable
    intervals: dict[str, NDArray[np.float64]]
    allocations: list[int]
    stopping_reasons: list[str]
    feasibility: list[str]
    policy: ReplicationPolicy


@dataclass
class _Allocation:
    trials: list[list[TrialResult]]
    reasons: list[str]
    feasibility: list[str]
    evaluations: int = 0

    @property
    def counts(self) -> NDArray[np.int64]:
        return np.array([len(rows) for rows in self.trials], dtype=np.int64)


def _validate(
    policy: ReplicationPolicy,
    observables: list[Observable],
    constraints: list[Constraint],
    n_designs: int,
) -> None:
    names = {o.name for o in observables}
    if not names or len(names) != len(observables):
        msg = "Sequential replication requires designs and distinct observables"
        raise ValueError(msg)
    if policy.min_reps < 1 or policy.max_reps < policy.min_reps:
        msg = "Require 1 <= min_reps <= max_reps"
        raise ValueError(msg)
    if policy.max_evaluations < n_designs * policy.min_reps:
        msg = "max_evaluations must cover min_reps for every design"
        raise ValueError(msg)
    if not 0 < policy.confidence < 1:
        msg = "confidence must be strictly between zero and one"
        raise ValueError(msg)
    if set(policy.score_bounds) != names:
        msg = "score_bounds must specify every observable exactly"
        raise ValueError(msg)
    for name, bounds in policy.score_bounds.items():
        if len(bounds) != 2 or not np.all(np.isfinite(bounds)) or bounds[0] > bounds[1]:
            msg = f"Invalid score bounds for {name!r}"
            raise ValueError(msg)
    if set(policy.equivalence) - names or any(
        not np.isfinite(v) or v < 0 for v in policy.equivalence.values()
    ):
        msg = "equivalence requires nonnegative finite tolerances for known observables"
        raise ValueError(msg)
    _validate_constraints(constraints, names, policy.confidence)


def _validate_constraints(
    constraints: list[Constraint], names: set[str], confidence: float
) -> None:
    for constraint in constraints:
        if (
            constraint.observable not in names
            or constraint.op not in {"<", "<=", ">", ">="}
            or not np.isfinite(constraint.threshold)
            or (
                constraint.confidence is not None and constraint.confidence > confidence
            )
        ):
            msg = (
                "Constraints require a scored observable, monotone operator "
                "and compatible confidence"
            )
            raise ValueError(msg)


def _evaluate(
    allocation: _Allocation,
    world: Simulator,
    scorer: Scorer,
    config: dict[str, Any],
    design_point: int,
    policy: ReplicationPolicy,
) -> None:
    rep = len(allocation.trials[design_point])
    trial = _run_single(
        world, scorer, config, rep=rep, supports_rep=_generate_accepts_rep(world)
    )
    for name, (lower, upper) in policy.score_bounds.items():
        value = trial.scores.get(name, float("nan"))
        if not np.isfinite(value) or not lower <= value <= upper:
            msg = (
                f"Score {name!r}={value} violates its declared bounds {(lower, upper)}"
            )
            raise ValueError(msg)
    trial.metadata.update({
        "design_point": design_point,
        "allocation": "initial" if rep < policy.min_reps else "ambiguous",
    })
    allocation.trials[design_point].append(trial)
    allocation.evaluations += 1


def _intervals(
    allocation: _Allocation, observables: list[Observable], policy: ReplicationPolicy
) -> NDArray[np.float64]:
    means = np.array([
        [np.mean([t.scores[o.name] for t in rows]) for o in observables]
        for rows in allocation.trials
    ])
    bounds = np.array([policy.score_bounds[o.name] for o in observables])
    multiplicity = len(allocation.trials) * len(observables) * policy.max_reps
    radius = (bounds[:, 1] - bounds[:, 0]) * np.sqrt(
        np.log(2 * multiplicity / (1 - policy.confidence))
        / (2 * allocation.counts[:, None])
    )
    return np.stack(
        [
            np.maximum(means - radius, bounds[:, 0]),
            np.minimum(means + radius, bounds[:, 1]),
        ],
        axis=2,
    )


def _feasibility(
    intervals: NDArray[np.float64], constraints: list[Constraint], names: list[str]
) -> list[str]:
    states = []
    for row in intervals:
        state = "feasible"
        for constraint in constraints:
            lower, upper = row[names.index(constraint.observable)]
            threshold = constraint.threshold
            tests = {
                "<=": (upper <= threshold, lower > threshold),
                "<": (upper < threshold, lower >= threshold),
                ">=": (lower >= threshold, upper < threshold),
                ">": (lower > threshold, upper <= threshold),
            }
            feasible, infeasible = tests[constraint.op]
            if infeasible:
                state = "infeasible"
                break
            if not feasible:
                state = "unknown"
        states.append(state)
    return states


def _classify(
    allocation: _Allocation,
    intervals: NDArray[np.float64],
    observables: list[Observable],
    constraints: list[Constraint],
    policy: ReplicationPolicy,
) -> list[int]:
    allocation.feasibility = _feasibility(
        intervals, constraints, [o.name for o in observables]
    )
    signs = np.array([
        1 if o.direction == Direction.MINIMIZE else -1 for o in observables
    ])
    oriented = np.sort(intervals * signs[None, :, None], axis=2)
    for index, reason in enumerate(allocation.reasons):
        if reason:
            continue
        if allocation.feasibility[index] == "infeasible":
            allocation.reasons[index] = "confidently_infeasible"
        elif any(
            other != index
            and allocation.feasibility[other] == "feasible"
            and np.all(oriented[other, :, 1] <= oriented[index, :, 0])
            and np.any(oriented[other, :, 1] < oriented[index, :, 0])
            for other in range(len(allocation.trials))
        ):
            allocation.reasons[index] = "confidently_dominated"
    active = [i for i, reason in enumerate(allocation.reasons) if not reason]
    if active and all(allocation.feasibility[i] == "feasible" for i in active):
        _resolve_front(allocation, intervals, oriented, active, observables, policy)
    return [i for i in active if not allocation.reasons[i]]


def _resolve_front(
    allocation: _Allocation,
    intervals: NDArray[np.float64],
    oriented: NDArray[np.float64],
    active: list[int],
    observables: list[Observable],
    policy: ReplicationPolicy,
) -> None:
    tolerance = np.array([policy.equivalence.get(o.name, 0) for o in observables])
    equivalent = np.all(
        intervals[active, :, 1].max(axis=0) - intervals[active, :, 0].min(axis=0)
        <= tolerance
    )
    resolved = all(
        np.any(oriented[a, :, 1] < oriented[b, :, 0])
        and np.any(oriented[b, :, 1] < oriented[a, :, 0])
        for a in active
        for b in active
        if a != b
    )
    reason = (
        "selected"
        if len(active) == 1
        else "practically_equivalent"
        if equivalent
        else "resolved_front"
        if resolved
        else ""
    )
    if reason:
        for index in active:
            allocation.reasons[index] = reason


def _result(
    allocation: _Allocation,
    intervals: NDArray[np.float64],
    observables: list[Observable],
    policy: ReplicationPolicy,
) -> SequentialResult:
    names = [o.name for o in observables]
    rows = [t for trials in allocation.trials for t in trials]
    raw = ResultsTable(
        configs=[t.config for t in rows],
        scores=np.array(
            [[t.scores[name] for name in names] for t in rows], dtype=float
        ),
        observable_names=names,
        metadata=[{"wall_seconds": t.wall_seconds, **t.metadata} for t in rows],
    )
    summary = raw.aggregate_replicates()
    for index, meta in enumerate(summary.metadata):
        meta.update({
            "stopping_reason": allocation.reasons[index],
            "feasibility": allocation.feasibility[index],
            "intervals": {
                name: intervals[index, j].tolist() for j, name in enumerate(names)
            },
            "confidence": policy.confidence,
            "interval_method": "finite_horizon_hoeffding_union_bound",
        })
    return SequentialResult(
        raw,
        summary,
        {name: intervals[:, j, :] for j, name in enumerate(names)},
        allocation.counts.tolist(),
        allocation.reasons,
        allocation.feasibility,
        policy,
    )


def run_sequential(
    world: Simulator,
    scorer: Scorer,
    grid: list[dict[str, Any]],
    observables: list[Observable],
    *,
    policy: ReplicationPolicy,
    constraints: list[Constraint] | None = None,
) -> SequentialResult:
    """Allocate extra replicates to unresolved comparisons and constraints.

    Args:
        world: Simulator, with optional replicate-id support as in run_grid.
        scorer: Scorer returning finite values within declared score bounds.
        grid: Fixed candidate designs; adaptive proposal selection is excluded.
        observables: Objectives with directions; scores and bounds use raw units.
        policy: Explicit budgets, score bounds, joint confidence and equivalence.
        constraints: Monotone feasibility constraints on scored observables.
            Simultaneous policy intervals replace normal standard-error bounds.

    Returns:
        Raw per-replicate results, per-design means, simultaneous mean intervals,
        allocations and stopping reasons. Unresolved cases retain unknown
        feasibility and stop at max_reps or the total evaluation budget.

    Raises:
        ValueError: If definitions/budgets are invalid or a score violates bounds.

    Notes:
        Coverage requires independent replicates with a fixed mean within each
        design and known almost-sure score bounds. Dependence across designs
        through common random numbers is allowed. Intervals use Hoeffding bounds
        and a union bound over all designs, observables and sample sizes through
        max_reps, permitting data-dependent allocation and stopping. These
        conservative intervals do not turn ordinary post-selection paired
        comparisons into sequentially valid inference.
    """
    if not grid:
        msg = "Sequential replication requires at least one design"
        raise ValueError(msg)
    constraints = list(constraints or [])
    _validate(policy, observables, constraints, len(grid))
    allocation = _Allocation(
        [[] for _ in grid], [""] * len(grid), ["unknown"] * len(grid)
    )
    for _rep in range(policy.min_reps):
        for index, config in enumerate(grid):
            _evaluate(allocation, world, scorer, config, index, policy)
    intervals = _intervals(allocation, observables, policy)
    active = _classify(allocation, intervals, observables, constraints, policy)
    while active:
        eligible = [i for i in active if len(allocation.trials[i]) < policy.max_reps]
        if not eligible or allocation.evaluations >= policy.max_evaluations:
            for index in active:
                allocation.reasons[index] = (
                    "max_reps"
                    if len(allocation.trials[index]) >= policy.max_reps
                    else "budget_exhausted"
                )
            break
        index = min(eligible, key=lambda i: (len(allocation.trials[i]), i))
        _evaluate(allocation, world, scorer, grid[index], index, policy)
        intervals = _intervals(allocation, observables, policy)
        active = _classify(allocation, intervals, observables, constraints, policy)
    return _result(allocation, intervals, observables, policy)
