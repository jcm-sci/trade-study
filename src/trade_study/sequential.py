"""Budgeted replication with simultaneous finite-horizon confidence bounds."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
from numbers import Integral
from typing import TYPE_CHECKING, Any

import numpy as np

from .protocols import Direction, ResultsTable
from .runner import _generate_accepts_rep, _run_single

if TYPE_CHECKING:
    from typing import Literal

    from numpy.typing import NDArray

    from .protocols import Constraint, Observable, Scorer, Simulator, TrialResult


@dataclass(frozen=True)
class ReplicationPolicy:
    """Explicit budget and bounded-score assumptions for sequential replication.

    Attributes:
        score_bounds: Known almost-sure bounds in raw observable units.
        min_reps: Initial adaptive batch per design; must not exceed max_reps.
        max_reps: Adaptive per-design maximum/horizon, or exact fixed-plan count.
        max_evaluations: Total simulator/scorer evaluations allowed.
        confidence: Joint coverage across all designs, observables and looks.
        equivalence: Practical-equivalence tolerances in raw observable units;
            omitted observables use zero.
        allocation: Adaptive retirement/allocation, or a fixed plan collecting
            exactly max_reps for every design before any inference look.
        replicate_unit: Independent simulator/scorer evaluations, or independent
            cluster scores. Cluster members are not additional replicates.
    """

    score_bounds: dict[str, tuple[float, float]]
    min_reps: int = 2
    max_reps: int = 100
    max_evaluations: int = 1000
    confidence: float = 0.95
    equivalence: dict[str, float] = field(default_factory=dict)
    allocation: Literal["adaptive", "fixed"] = "adaptive"
    replicate_unit: Literal["independent_replicate", "independent_cluster"] = (
        "independent_replicate"
    )


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


@dataclass(frozen=True)
class SequentialFailure:
    """Detached diagnostics from an aborted run, without inferential results.

    Attributes:
        results: Completed raw rows, each marked as belonging to an aborted run.
        grid: Declared designs captured before evaluation.
        observables: Declared objectives captured before evaluation.
        constraints: Declared constraints captured before evaluation.
        policy: Detached policy used by the failed run.
        completed_reps: Completed counts in declared design order.
        failed_design: Zero-based design index of the failed attempt.
        failed_rep: Zero-based replicate index of the failed attempt.
        stage: Evaluation or declared-score validation stage.
        exception_type: Qualified type of the original evaluation error.
        exception_message: Original error message.
    """

    results: ResultsTable
    grid: list[dict[str, Any]]
    observables: list[Observable]
    constraints: list[Constraint]
    policy: ReplicationPolicy
    completed_reps: tuple[int, ...]
    failed_design: int
    failed_rep: int
    stage: Literal["evaluation", "score_validation"]
    exception_type: str
    exception_message: str

    def to_metadata(self) -> dict[str, Any]:
        """Describe the abort for caller provenance in a result bundle.

        Returns:
            Versioned diagnostic metadata. Designs must also satisfy the bundle
            writer's primitive-JSON contract when persisted. Counts distinguish
            failed attempts from completed rows and unattempted capacity.
        """
        failed = [int(index == self.failed_design) for index in range(len(self.grid))]
        attempted = [
            count + failures
            for count, failures in zip(self.completed_reps, failed, strict=True)
        ]
        return {
            "schema_version": 1,
            "kind": "sequential_failure",
            "status": "aborted",
            "evaluability": "unavailable",
            "decision_status": "unavailable",
            "inference": "not_supplied",
            "failure_policy": "abort",
            "designs": deepcopy(self.grid),
            "completed_reps": list(self.completed_reps),
            "attempted_reps": attempted,
            "failed_reps": failed,
            "actual_evaluations": sum(attempted),
            "completed_evaluations": sum(self.completed_reps),
            "failed_design": self.failed_design,
            "failed_rep": self.failed_rep,
            "failed_evaluation_index": sum(self.completed_reps),
            "stage": self.stage,
            "error_type": self.exception_type,
            "error_message": self.exception_message,
            "allocation_policy": self.policy.allocation,
            "replicate_unit": self.policy.replicate_unit,
            "planned_min_reps": self.policy.min_reps,
            "planned_reps": self.policy.max_reps
            if self.policy.allocation == "fixed"
            else None,
            "max_reps": self.policy.max_reps,
            "max_evaluations": self.policy.max_evaluations,
            "unattempted_capacity": [self.policy.max_reps - n for n in attempted],
            "requested_confidence": self.policy.confidence,
            "score_bounds": {
                name: [lower, upper]
                for name, (lower, upper) in self.policy.score_bounds.items()
            },
            "equivalence": dict(self.policy.equivalence),
            "observables": [
                {
                    "name": item.name,
                    "direction": item.direction.value,
                    "weight": float(item.weight),
                }
                for item in self.observables
            ],
            "constraints": [
                {
                    "name": item.name,
                    "observable": item.observable,
                    "op": item.op,
                    "threshold": float(item.threshold),
                    "requested_confidence": None
                    if item.confidence is None
                    else float(item.confidence),
                }
                for item in self.constraints
            ],
        }


class SequentialEvaluationError(RuntimeError):
    """An opt-in aborted-run error carrying diagnostics and its original cause.

    Attributes:
        failure: Detached raw rows and failure metadata, without intervals,
            summary feasibility or a completed sequential result.
    """

    def __init__(self, failure: SequentialFailure) -> None:
        """Store diagnostics for an explicitly requested failure capture."""
        self.failure = failure
        super().__init__(
            f"Sequential evaluation aborted at design {failure.failed_design}, "
            f"replicate {failure.failed_rep}: {failure.exception_type}: "
            f"{failure.exception_message}"
        )


@dataclass(frozen=True)
class _FailureCapture:
    grid: list[dict[str, Any]]
    observables: list[Observable]
    constraints: list[Constraint]


@dataclass
class _Allocation:
    trials: list[list[TrialResult]]
    reasons: list[str]
    feasibility: list[str]
    evaluations: int = 0
    failure_capture: _FailureCapture | None = None

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
    _validate_allocation(policy, n_designs)
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


def _validate_allocation(policy: ReplicationPolicy, n_designs: int) -> None:
    """Require a supported allocation and complete prespecified integer budget.

    Raises:
        ValueError: Counts, allocation, replicate unit or available budget are invalid.
    """
    counts = (policy.min_reps, policy.max_reps, policy.max_evaluations)
    if any(
        isinstance(value, bool) or not isinstance(value, Integral) for value in counts
    ):
        msg = "min_reps, max_reps and max_evaluations must be integers, not booleans"
        raise ValueError(msg)
    if policy.min_reps < 1 or policy.max_reps < policy.min_reps:
        msg = "Require 1 <= min_reps <= max_reps"
        raise ValueError(msg)
    if policy.max_evaluations < n_designs * policy.min_reps:
        msg = "max_evaluations must cover min_reps for every design"
        raise ValueError(msg)
    if policy.allocation not in {"adaptive", "fixed"} or policy.replicate_unit not in {
        "independent_replicate",
        "independent_cluster",
    }:
        msg = "Unsupported allocation or replicate_unit policy"
        raise ValueError(msg)
    if (
        policy.allocation == "fixed"
        and policy.max_evaluations < n_designs * policy.max_reps
    ):
        msg = (
            "Fixed allocation requires max_evaluations to cover every planned replicate"
        )
        raise ValueError(msg)


def _snapshot_policy(policy: ReplicationPolicy) -> ReplicationPolicy:
    """Detach the validated run definition from caller-owned mutable mappings.

    Returns:
        Used policy with detached bounds/tolerances and primitive numeric fields.
    """
    return replace(
        policy,
        score_bounds={
            name: (float(lower), float(upper))
            for name, (lower, upper) in policy.score_bounds.items()
        },
        equivalence={name: float(value) for name, value in policy.equivalence.items()},
        min_reps=int(policy.min_reps),
        max_reps=int(policy.max_reps),
        max_evaluations=int(policy.max_evaluations),
        confidence=float(policy.confidence),
    )


def _looks(policy: ReplicationPolicy) -> int:
    """Return the prespecified number of inference looks per score stream."""
    return 1 if policy.allocation == "fixed" else policy.max_reps


def _interval_method(policy: ReplicationPolicy) -> str:
    """Return the method name for the actually used look family."""
    return (
        "fixed_replicates_hoeffding_union_bound"
        if policy.allocation == "fixed"
        else "finite_horizon_hoeffding_union_bound"
    )


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
    stage: Literal["evaluation", "score_validation"] = "evaluation"
    try:
        trial = _run_single(
            world, scorer, config, rep=rep, supports_rep=_generate_accepts_rep(world)
        )
        stage = "score_validation"
        _validate_trial_scores(trial, policy)
    # Evaluators may raise any ordinary Exception; default behavior re-raises it.
    except Exception as error:
        if allocation.failure_capture is None:
            raise
        raise SequentialEvaluationError(
            _failure(
                allocation.failure_capture,
                allocation,
                policy,
                design_point,
                rep,
                stage,
                error,
            )
        ) from error
    trial.metadata.update({
        "design_point": design_point,
        "allocation": "fixed"
        if policy.allocation == "fixed"
        else "initial"
        if rep < policy.min_reps
        else "ambiguous",
    })
    if allocation.failure_capture is not None:
        trial.metadata["evaluation_index"] = allocation.evaluations
    allocation.trials[design_point].append(trial)
    allocation.evaluations += 1


def _validate_trial_scores(trial: TrialResult, policy: ReplicationPolicy) -> None:
    """Check every declared score before accepting a completed replicate.

    Raises:
        ValueError: A declared score is missing, nonfinite or out of bounds.
    """
    for name, (lower, upper) in policy.score_bounds.items():
        value = trial.scores.get(name, float("nan"))
        if not np.isfinite(value) or not lower <= value <= upper:
            msg = (
                f"Score {name!r}={value} violates its declared bounds {(lower, upper)}"
            )
            raise ValueError(msg)


def _intervals(
    allocation: _Allocation, observables: list[Observable], policy: ReplicationPolicy
) -> NDArray[np.float64]:
    means = np.array([
        [np.mean([t.scores[o.name] for t in rows]) for o in observables]
        for rows in allocation.trials
    ])
    bounds = np.array([policy.score_bounds[o.name] for o in observables])
    multiplicity = len(allocation.trials) * len(observables) * _looks(policy)
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
    constraints: list[Constraint],
) -> SequentialResult:
    names = [o.name for o in observables]
    raw = _raw_results(allocation, observables)
    summary = raw.aggregate_replicates()
    method = _interval_method(policy)
    for index, meta in enumerate(summary.metadata):
        meta.update({
            "stopping_reason": allocation.reasons[index],
            "feasibility": allocation.feasibility[index],
            "intervals": {
                name: intervals[index, j].tolist() for j, name in enumerate(names)
            },
            "confidence": policy.confidence,
            "interval_method": method,
            "replication_contract": _contract_metadata(
                allocation,
                intervals[index],
                observables,
                policy,
                index,
                constraints,
            ),
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


def _raw_results(
    allocation: _Allocation, observables: list[Observable]
) -> ResultsTable:
    """Collect raw completed rows.

    Returns:
        A score table with explicit row/observable axes, including zero rows.
    """
    names = [item.name for item in observables]
    rows = [trial for trials in allocation.trials for trial in trials]
    return ResultsTable(
        configs=[trial.config for trial in rows],
        scores=np.array(
            [[trial.scores[name] for name in names] for trial in rows], dtype=float
        ).reshape(len(rows), len(names)),
        observable_names=names,
        metadata=[
            {"wall_seconds": trial.wall_seconds, **trial.metadata} for trial in rows
        ],
    )


def _failure(
    capture: _FailureCapture,
    allocation: _Allocation,
    policy: ReplicationPolicy,
    design_point: int,
    rep: int,
    stage: Literal["evaluation", "score_validation"],
    error: Exception,
) -> SequentialFailure:
    """Detach completed rows and the declared plan without computing inference.

    Returns:
        An aborted-run artifact with the failed attempt retained separately.
    """
    raw = deepcopy(_raw_results(allocation, capture.observables))
    for metadata in raw.metadata:
        metadata.update(run_status="aborted", decision_status="unavailable")
    return SequentialFailure(
        results=raw,
        grid=deepcopy(capture.grid),
        observables=deepcopy(capture.observables),
        constraints=deepcopy(capture.constraints),
        policy=_snapshot_policy(policy),
        completed_reps=tuple(int(count) for count in allocation.counts),
        failed_design=design_point,
        failed_rep=rep,
        stage=stage,
        exception_type=f"{type(error).__module__}.{type(error).__qualname__}",
        exception_message=str(error),
    )


def _contract_metadata(
    allocation: _Allocation,
    intervals: NDArray[np.float64],
    observables: list[Observable],
    policy: ReplicationPolicy,
    index: int,
    constraints: list[Constraint],
) -> dict[str, Any]:
    """Record the used inference family separately from power and gate errors.

    Returns:
        Versioned primitive metadata for completed, fully evaluable runs. A failed
        evaluation aborts rather than returning a reduced-denominator success.
    """
    names = [observable.name for observable in observables]
    multiplicity = len(allocation.trials) * len(names) * _looks(policy)
    return {
        "schema_version": 1,
        "allocation_policy": policy.allocation,
        "planned_min_reps": policy.min_reps,
        "planned_reps": policy.max_reps if policy.allocation == "fixed" else None,
        "max_reps": policy.max_reps,
        "actual_reps": len(allocation.trials[index]),
        "max_evaluations": policy.max_evaluations,
        "actual_evaluations": allocation.evaluations,
        "stopping_reason": allocation.reasons[index],
        "replicate_unit": policy.replicate_unit,
        "estimand": "mean_score_per_declared_replicate",
        "confidence_scope": "joint_design_observable_look_family",
        "confidence": policy.confidence,
        "interval_method": _interval_method(policy),
        "objectives": [
            {"name": observable.name, "direction": observable.direction.value}
            for observable in observables
        ],
        "constraints": [
            {
                "name": constraint.name,
                "observable": constraint.observable,
                "op": constraint.op,
                "threshold": float(constraint.threshold),
                "requested_confidence": None
                if constraint.confidence is None
                else float(constraint.confidence),
            }
            for constraint in constraints
        ],
        "constraint_interval_policy": "joint_policy_intervals",
        "n_designs": len(allocation.trials),
        "n_observables": len(names),
        "max_looks_per_stream": _looks(policy),
        "multiplicity": multiplicity,
        "per_look_error": (1.0 - policy.confidence) / multiplicity,
        "interval_half_widths": {
            name: float((bounds[1] - bounds[0]) / 2)
            for name, bounds in zip(names, intervals, strict=True)
        },
        "gate_power": {"status": "not_assessed", "value": None},
        "gate_false_failure_rate": {"status": "not_assessed", "value": None},
        "failure_policy": "abort",
        "failed_reps": 0,
        "evaluability": "available",
        "decision_status": "not_assessed"
        if policy.allocation == "fixed"
        else "unresolved"
        if allocation.reasons[index] in {"max_reps", "budget_exhausted"}
        else "resolved",
        "score_bounds": {
            name: [lower, upper] for name, (lower, upper) in policy.score_bounds.items()
        },
        "equivalence": dict(policy.equivalence),
        "assumptions": [
            "fixed_candidate_family",
            "independent_replicates_with_fixed_mean_within_design",
            "prespecified_almost_sure_score_bounds",
            "no_cross_design_independence_required",
        ],
        "paired_contrast_coverage": "not_supplied",
    }


def _run_fixed(
    world: Simulator,
    scorer: Scorer,
    grid: list[dict[str, Any]],
    observables: list[Observable],
    policy: ReplicationPolicy,
    constraints: list[Constraint],
    failure_capture: _FailureCapture | None = None,
) -> SequentialResult:
    """Execute the entire prespecified plan before its only inference look.

    Returns:
        A completed fixed-count baseline; no early winner/retirement is inferred.
    """
    allocation = _Allocation(
        [[] for _ in grid],
        ["fixed_plan_complete"] * len(grid),
        ["unknown"] * len(grid),
        failure_capture=failure_capture,
    )
    for _rep in range(policy.max_reps):
        for index, config in enumerate(grid):
            _evaluate(allocation, world, scorer, config, index, policy)
    intervals = _intervals(allocation, observables, policy)
    allocation.feasibility = _feasibility(
        intervals, constraints, [observable.name for observable in observables]
    )
    return _result(allocation, intervals, observables, policy, constraints)


def run_sequential(
    world: Simulator,
    scorer: Scorer,
    grid: list[dict[str, Any]],
    observables: list[Observable],
    *,
    policy: ReplicationPolicy,
    constraints: list[Constraint] | None = None,
    capture_failure: bool = False,
) -> SequentialResult:
    """Allocate extra replicates to unresolved comparisons and constraints.

    Args:
        world: Simulator, with optional replicate-id support as in run_grid.
        scorer: Scorer returning finite values within declared score bounds.
        grid: Fixed candidate designs; adaptive proposal selection is excluded.
        observables: Objectives with directions; scores and bounds use raw units.
        policy: Explicit budgets, bounds, confidence and adaptive/fixed allocation.
            The validated used policy is snapshotted before simulator calls.
        constraints: Monotone feasibility constraints on scored observables.
            Simultaneous policy intervals replace normal standard-error bounds.
        capture_failure: On ordinary evaluation/score-validation exceptions, raise
            SequentialEvaluationError with detached raw diagnostics and the
            original exception as its cause. No successful result is returned.

    Returns:
        Raw per-replicate results, per-design means, simultaneous mean intervals,
        allocations and stopping reasons. Unresolved cases retain unknown
        feasibility and stop at max_reps or the total evaluation budget.

    Raises:
        ValueError: If definitions/budgets are invalid or a score violates bounds.
        TypeError: If capture_failure is not boolean.
        SequentialEvaluationError: If explicitly requested failure capture aborts
            an evaluation. Its failure record supplies no inferential result.

    Notes:
        Coverage requires independent replicates with a fixed mean within each
        design and known almost-sure score bounds. Dependence across designs
        through common random numbers is allowed. Adaptive intervals use Hoeffding
        bounds and a union bound over all designs, observables and sample sizes
        through max_reps, permitting data-dependent allocation and stopping.
        Fixed mode collects every planned replicate and uses one final look;
        those intervals supply no interim-look guarantee. Independent clusters
        can supply one score per replicate, without counting members separately.
        Gate power and false-failure rates are not assessed by these intervals.
        Missing/non-finite/out-of-bound scores abort rather than reduce the
        denominator. Returned metadata records the used policy and actual counts;
        ordinary post-selection paired intervals gain no sequential coverage.
    """  # ruff: ignore[docstring-extraneous-exception] -- helper propagates error
    if not isinstance(capture_failure, bool):
        msg = "capture_failure must be boolean"
        raise TypeError(msg)
    grid, observables = list(grid), list(observables)
    if not grid:
        msg = "Sequential replication requires at least one design"
        raise ValueError(msg)
    constraints = list(constraints or [])
    _validate(policy, observables, constraints, len(grid))
    policy = _snapshot_policy(policy)
    capture = (
        _FailureCapture(deepcopy(grid), deepcopy(observables), deepcopy(constraints))
        if capture_failure
        else None
    )
    if policy.allocation == "fixed":
        return _run_fixed(
            world, scorer, grid, observables, policy, constraints, capture
        )
    allocation = _Allocation(
        [[] for _ in grid],
        [""] * len(grid),
        ["unknown"] * len(grid),
        failure_capture=capture,
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
    return _result(allocation, intervals, observables, policy, constraints)
