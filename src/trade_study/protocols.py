"""Core protocols and data types for model criticism studies."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


class Tier(Enum):
    """Observable classification tier (MFAI §4 hierarchy)."""

    EMBEDDED = "embedded"
    PENALIZED = "penalized"
    DIAGNOSTIC = "diagnostic"
    COST = "cost"


class Direction(Enum):
    """Optimization direction for an observable."""

    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


@dataclass(frozen=True)
class Observable:
    """A structured observable evaluated against known truth.

    Attributes:
        name: Identifier (e.g. "coverage_95", "relWIS", "wall_seconds").
        tier: Classification in the embedded/penalized/diagnostic/cost hierarchy.
        direction: Whether lower or higher values are better.
    """

    name: str
    tier: Tier
    direction: Direction


@runtime_checkable
class ModelWorld(Protocol):
    """Protocol for generating ground truth and observations.

    A model world produces (truth, observations) pairs where truth is the
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
            truth: Known latent state from the model world.
            observations: Observed data from the model world.
            config: The configuration that produced this trial.

        Returns:
            Dictionary mapping observable names to scalar scores.
        """
        ...


@dataclass
class TrialResult:
    """Result of a single model world trial."""

    config: dict[str, Any]
    scores: dict[str, float]
    wall_seconds: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Annotation:
    """External information attached to configurations.

    Used for costs, constraints, or metadata not computed by the model world
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
