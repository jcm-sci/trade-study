"""Multi-objective trade-study orchestration.

Scoring, Pareto optimization, and Bayesian stacking.
"""

from ._pareto import extract_front, hypervolume, igd_plus, pareto_rank
from ._scoring import coverage_curve, score
from ._version import __version__
from .design import Factor, FactorType, build_grid, reduce_factors, screen
from .io import load_results, save_results
from .protocols import (
    Annotation,
    Direction,
    Observable,
    ResultsTable,
    Scorer,
    Simulator,
    TrialResult,
)
from .runner import run_adaptive, run_grid
from .stacking import ensemble_predict, stack_bayesian, stack_scores
from .study import Phase, Study, top_k_pareto_filter, weighted_sum_filter
from .viz import plot_calibration, plot_front, plot_parallel, plot_scores

__all__ = [
    "Annotation",
    "Direction",
    "Factor",
    "FactorType",
    "Observable",
    "Phase",
    "ResultsTable",
    "Scorer",
    "Simulator",
    "Study",
    "TrialResult",
    "__version__",
    "build_grid",
    "coverage_curve",
    "ensemble_predict",
    "extract_front",
    "hypervolume",
    "igd_plus",
    "load_results",
    "pareto_rank",
    "plot_calibration",
    "plot_front",
    "plot_parallel",
    "plot_scores",
    "reduce_factors",
    "run_adaptive",
    "run_grid",
    "save_results",
    "score",
    "screen",
    "stack_bayesian",
    "stack_scores",
    "top_k_pareto_filter",
    "weighted_sum_filter",
]
