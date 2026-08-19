"""Core protocols and data types for model criticism studies."""

from __future__ import annotations

import operator as _operator
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np
    from numpy.typing import NDArray


class Direction(Enum):
    """Optimization direction for an observable."""

    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


@dataclass(frozen=True)
class Observable:
    """A structured observable evaluated against known truth.

    Attributes:
        name: Identifier (e.g. "coverage_95", "relWIS", "wall_seconds").
        direction: Whether lower or higher values are better.
        weight: Relative importance for weighted Pareto analysis.
            Default ``1.0`` preserves unweighted behavior.
    """

    name: str
    direction: Direction
    weight: float = 1.0


_OP_MAP: dict[str, Callable[[Any, Any], bool]] = {
    ">=": _operator.ge,
    "<=": _operator.le,
    ">": _operator.gt,
    "<": _operator.lt,
    "==": _operator.eq,
    "!=": _operator.ne,
}


@dataclass(frozen=True)
class Constraint:
    """Feasibility constraint on an observable or annotation.

    A design is feasible when ``scores[observable] <op> threshold`` is
    true.

    Attributes:
        name: Human-readable label (e.g. ``"min_conversion"``).
        observable: Name of the observable or annotation column to test.
        op: Comparison operator as a string (``">="`` ``"<="`` ``">"``
            ``"<"`` ``"=="`` ``"!="``).
        threshold: Scalar threshold value.
    """

    name: str
    observable: str
    op: str
    threshold: float

    def __post_init__(self) -> None:
        """Validate the comparison operator.

        Raises:
            ValueError: If *op* is not one of the supported operators.
        """
        if self.op not in _OP_MAP:
            msg = (
                f"Constraint {self.name!r}: unsupported operator {self.op!r}. "
                f"Use one of {sorted(_OP_MAP)}"
            )
            raise ValueError(msg)

    def check(self, value: float) -> bool:
        """Test whether a scalar value satisfies the constraint.

        Args:
            value: Scalar score or annotation value to test.

        Returns:
            ``True`` if the value satisfies the constraint.
        """
        return bool(_OP_MAP[self.op](value, self.threshold))


@runtime_checkable
class Simulator(Protocol):
    """Protocol for generating ground truth and observations.

    A simulator produces (truth, observations) pairs where truth is the
    known latent state and observations are what a real system would see.
    """

    def generate(self, config: dict[str, Any]) -> tuple[Any, Any]:
        """Generate a (truth, observations) pair for a given configuration.

        Args:
            config: Dictionary of factor values defining this trial.

        Returns:
            A tuple of (truth, observations) where truth is the known latent
            state and observations are the (possibly noisy/masked) data.

        Note:
            Implementations that want independent draws per replicate under
            ``run_grid(..., n_reps>1)`` (#112) may additionally accept an
            optional keyword-only ``rep: int`` parameter (e.g.
            ``def generate(self, config, *, rep=0)``) and vary their own
            randomness by it (a seed derived from ``rep``, a per-rep RNG,
            etc.). :func:`~trade_study.runner.run_grid` detects this via
            introspection and passes the current 0-indexed replicate;
            simulators without a ``rep`` parameter are called unchanged and
            simply produce identical replicates, matching pre-#112
            behavior.
        """
        ...


@runtime_checkable
class PartialEvaluator(Protocol):
    """Protocol for incrementally-evaluable trials.

    Used by successive-halving / Hyperband (#104) to discard unpromising
    configurations after a small fraction of their full budget. The budget
    is opaque to the runner — it may be epochs, MCMC iterations, dataset
    fractions, mesh resolutions, or seconds. Implementations should
    interpret ``budget`` as "run from scratch up to this much work" so a
    trial promoted from rung *r* to rung *r+1* is re-trained at the larger
    budget rather than continuing from the smaller one (this matches the
    canonical Hyperband formulation; implementations are free to cache
    intermediate state internally as an optimization).
    """

    def evaluate(
        self,
        config: dict[str, Any],
        budget: float,
    ) -> dict[str, float]:
        """Evaluate ``config`` at the given ``budget`` and return observables.

        Args:
            config: Dictionary of factor values defining this trial.
            budget: Resource budget (epochs, iterations, dataset fraction,
                wall seconds, ...). Larger means a higher-fidelity
                evaluation.

        Returns:
            Mapping from observable name to scalar value, including the
            metric used for early-stopping.
        """
        ...


@runtime_checkable
class Scorer(Protocol):
    """Protocol for scoring model output against truth."""

    def score(
        self,
        truth: Any,
        observations: Any,
        config: dict[str, Any],
    ) -> dict[str, float]:
        """Score a single trial, returning values for each observable.

        Args:
            truth: Known latent state from the simulator.
            observations: Observed data from the simulator.
            config: The configuration that produced this trial.

        Returns:
            Dictionary mapping observable names to scalar scores.
        """
        ...


@dataclass
class TrialResult:
    """Result of a single simulation trial."""

    config: dict[str, Any]
    scores: dict[str, float]
    wall_seconds: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Annotation:
    """External information attached to configurations.

    Used for costs, constraints, or metadata not computed by the simulator
    (e.g. dollar costs from a surveillance costing sheet).

    Attributes:
        name: Column name in the results table.
        lookup: Dictionary mapping config key → value, or a callable.
        key: Which config field to use for lookup.
    """

    name: str
    lookup: dict[str, float] | Any
    key: str

    def resolve(self, config: dict[str, Any]) -> float:
        """Resolve the annotation value for a given config.

        Returns:
            The resolved annotation value as a float.
        """
        k = config[self.key]
        if callable(self.lookup):
            return float(self.lookup(k))
        return float(self.lookup[k])


@dataclass
class ResultsTable:
    """Scored results from a study phase.

    Stores configs, observable scores, annotations, and metadata
    as parallel arrays backed by numpy.
    """

    configs: list[dict[str, Any]]
    scores: NDArray[np.floating[Any]]  # (n_trials, n_observables)
    observable_names: list[str]
    annotations: NDArray[np.floating[Any]] | None = None  # (n_trials, n_annotations)
    annotation_names: list[str] = field(default_factory=list)
    metadata: list[dict[str, Any]] = field(default_factory=list)

    def feasible(self, constraints: list[Constraint]) -> NDArray[np.bool_]:
        """Return a boolean mask indicating which rows satisfy all constraints.

        Each constraint references an observable or annotation column by
        name.  A row is feasible only when **every** constraint evaluates
        to ``True``.

        Args:
            constraints: Constraint objects to evaluate.

        Returns:
            Boolean array of shape ``(n_trials,)``.

        Raises:
            KeyError: If a constraint references a column not found in
                either ``observable_names`` or ``annotation_names``.
        """
        import numpy as np

        mask = np.ones(len(self.configs), dtype=np.bool_)
        for con in constraints:
            if con.observable in self.observable_names:
                col_idx = self.observable_names.index(con.observable)
                values = self.scores[:, col_idx]
            elif (
                con.observable in self.annotation_names and self.annotations is not None
            ):
                col_idx = self.annotation_names.index(con.observable)
                values = self.annotations[:, col_idx]
            else:
                msg = (
                    f"Constraint {con.name!r}: column {con.observable!r} "
                    f"not found in observables or annotations"
                )
                raise KeyError(msg)
            mask &= _OP_MAP[con.op](values, con.threshold)
        return mask

    def aggregate_replicates(self) -> ResultsTable:
        """Collapse replicate rows into one row per design point (#112).

        Groups rows by their ``metadata["design_point"]`` key (set by
        :func:`~trade_study.runner.run_grid` when called with
        ``n_reps>1``) and averages scores within each group. Each
        aggregated row's metadata records ``design_point``, ``n_reps``
        (replicate count for that point), and ``score_std`` (per-observable
        standard deviation across replicates, keyed by observable name).
        Annotations, if present, are taken from the first replicate of each
        group (annotations are resolved from the config, which is identical
        across replicates of the same design point).

        Returns:
            A new ResultsTable with one row per unique design point.

        Raises:
            KeyError: If any row's metadata lacks a ``design_point`` key,
                i.e. this table wasn't produced by ``run_grid(n_reps>1)``.
        """
        import numpy as np

        groups: dict[int, list[int]] = {}
        for row_idx, meta in enumerate(self.metadata):
            if "design_point" not in meta:
                msg = (
                    "aggregate_replicates: row metadata is missing a "
                    "'design_point' key; only ResultsTables produced by "
                    "run_grid(..., n_reps>1) support this."
                )
                raise KeyError(msg)
            groups.setdefault(meta["design_point"], []).append(row_idx)

        order = sorted(groups)
        configs = [self.configs[groups[dp][0]] for dp in order]
        means = np.array([self.scores[groups[dp]].mean(axis=0) for dp in order])
        stds = np.array([self.scores[groups[dp]].std(axis=0, ddof=0) for dp in order])
        metadata = [
            {
                "design_point": dp,
                "n_reps": len(groups[dp]),
                "score_std": dict(zip(self.observable_names, stds[i], strict=True)),
            }
            for i, dp in enumerate(order)
        ]

        annotations = (
            np.array([self.annotations[groups[dp][0]] for dp in order])
            if self.annotations is not None
            else None
        )

        return ResultsTable(
            configs=configs,
            scores=means,
            observable_names=self.observable_names,
            annotations=annotations,
            annotation_names=self.annotation_names,
            metadata=metadata,
        )
