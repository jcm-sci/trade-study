"""Multi-objective trade-study orchestration: scoring, Pareto optimization, and Bayesian stacking."""

from ._version import __version__
from .protocols import (
    Annotation,
    Direction,
    ModelWorld,
    Observable,
    ResultsTable,
    Scorer,
    Tier,
    TrialResult,
)
from .study import Phase, Study, top_k_pareto_filter

__all__ = [
    "Annotation",
    "Direction",
    "ModelWorld",
    "Observable",
    "Phase",
    "ResultsTable",
    "Scorer",
    "Study",
    "Tier",
    "TrialResult",
    "__version__",
    "top_k_pareto_filter",
]
