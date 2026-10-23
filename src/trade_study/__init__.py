"""Multi-objective trade-study orchestration.

Scoring, Pareto optimization, and Bayesian stacking.
"""

from ._pareto import extract_front, hypervolume, igd_plus, pareto_rank
from ._scoring import coverage_curve, score
from ._version import __version__
from .artifacts import (
    ResultsBundle,
    load_result_bundle,
    publish_result_bundle,
    save_result_bundle,
)
from .cache import EvaluationCache
from .decision import PreferencePolicy, PreferenceSweep, preference_sweep
from .design import (
    Factor,
    FactorConstraint,
    FactorType,
    build_grid,
    reduce_factors,
    screen,
    sobol_indices,
    unit_to_value,
    value_to_unit,
)
from .fisher import (
    FisherContext,
    FisherScreenPolicy,
    FisherScreenResult,
    fisher_screen_counts,
    screen_fisher,
)
from .io import load_results, save_results
from .paired import PairedDifference, paired_difference, paired_rank
from .proportions import (
    ProportionInterval,
    proportion_interval,
    replicates_for_half_width,
)
from .protocols import (
    Annotation,
    Constraint,
    Direction,
    Observable,
    PartialEvaluator,
    ResultsTable,
    Scorer,
    Simulator,
    TrialResult,
)
from .regime import (
    RegimeSurrogate,
    aggregate_bucketed_config,
    fit_regime_surrogate,
    recommend_bucketed_config,
    recommend_per_regime,
)
from .runner import run_adaptive, run_grid, run_hyperband, run_successive_halving
from .sensitivity import TableSensitivity, sensitivity_from_table
from .sequential import (
    ReplicationPolicy,
    SequentialEvaluationError,
    SequentialFailure,
    SequentialResult,
    run_sequential,
)
from .session import AdaptiveSession, SessionTrial
from .slurm import (
    FrozenCheckout,
    SlurmPlan,
    SlurmResources,
    SlurmTask,
    freeze_checkout,
    prepare_slurm_plan,
)
from .stacking import ensemble_predict, stack_bayesian, stack_proportional, stack_scores
from .study import (
    Phase,
    Study,
    feasibility_filter,
    top_k_pareto_filter,
    weighted_sum_filter,
)
from .surrogate import PredictionSupport, SurrogateModel, fit_surrogate
from .viz import plot_calibration, plot_front, plot_parallel, plot_scores

__all__ = [
    "AdaptiveSession",
    "Annotation",
    "Constraint",
    "Direction",
    "EvaluationCache",
    "Factor",
    "FactorConstraint",
    "FactorType",
    "FisherContext",
    "FisherScreenPolicy",
    "FisherScreenResult",
    "FrozenCheckout",
    "Observable",
    "PairedDifference",
    "PartialEvaluator",
    "Phase",
    "PredictionSupport",
    "PreferencePolicy",
    "PreferenceSweep",
    "ProportionInterval",
    "RegimeSurrogate",
    "ReplicationPolicy",
    "ResultsBundle",
    "ResultsTable",
    "Scorer",
    "SequentialEvaluationError",
    "SequentialFailure",
    "SequentialResult",
    "SessionTrial",
    "Simulator",
    "SlurmPlan",
    "SlurmResources",
    "SlurmTask",
    "Study",
    "SurrogateModel",
    "TableSensitivity",
    "TrialResult",
    "__version__",
    "aggregate_bucketed_config",
    "build_grid",
    "coverage_curve",
    "ensemble_predict",
    "extract_front",
    "feasibility_filter",
    "fisher_screen_counts",
    "fit_regime_surrogate",
    "fit_surrogate",
    "freeze_checkout",
    "hypervolume",
    "igd_plus",
    "load_result_bundle",
    "load_results",
    "paired_difference",
    "paired_rank",
    "pareto_rank",
    "plot_calibration",
    "plot_front",
    "plot_parallel",
    "plot_scores",
    "preference_sweep",
    "prepare_slurm_plan",
    "proportion_interval",
    "publish_result_bundle",
    "recommend_bucketed_config",
    "recommend_per_regime",
    "reduce_factors",
    "replicates_for_half_width",
    "run_adaptive",
    "run_grid",
    "run_hyperband",
    "run_sequential",
    "run_successive_halving",
    "save_result_bundle",
    "save_results",
    "score",
    "screen",
    "screen_fisher",
    "sensitivity_from_table",
    "sobol_indices",
    "stack_bayesian",
    "stack_proportional",
    "stack_scores",
    "top_k_pareto_filter",
    "unit_to_value",
    "value_to_unit",
    "weighted_sum_filter",
]
