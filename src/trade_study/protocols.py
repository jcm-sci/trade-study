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
