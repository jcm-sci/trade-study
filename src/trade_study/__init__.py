"""Multi-objective trade-study orchestration.

Scoring, Pareto optimization, and Bayesian stacking.
"""

from ._version import __version__
from .protocols import (
    Annotation,
    Direction,
    Observable,
    ResultsTable,
    Scorer,
    Simulator,
    TrialResult,
)
from .study import Phase, Study, top_k_pareto_filter

__all__ = [
    "Annotation",
    "Direction",
    "Observable",
    "Phase",
    "ResultsTable",
    "Scorer",
    "Simulator",
    "Study",
    "TrialResult",
    "__version__",
    "top_k_pareto_filter",
]
