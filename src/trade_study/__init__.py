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
from .power import AdequateDesign, PowerCurve, minimum_adequate, power_curve
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
from .sequential import ReplicationPolicy, SequentialResult, run_sequential
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
from .trials import (
    AccrualPlan,
    EfficacyInterval,
    GroupSequentialDesign,
    OperatingCharacteristics,
    alpha_spending,
    conditional_power,
    design_group_sequential,
    efficacy_from_share,
    efficacy_interval,
    efficacy_share,
    events_required,
    expected_events_by_period,
    max_events_for_power,
    operating_characteristics,
    periods_to_events,
    unconditional_power,
)
from .viz import plot_calibration, plot_front, plot_parallel, plot_scores

__all__ = [
    "AccrualPlan",
    "AdaptiveSession",
    "AdequateDesign",
    "Annotation",
    "Constraint",
    "Direction",
    "EfficacyInterval",
    "EvaluationCache",
    "Factor",
    "FactorConstraint",
    "FactorType",
    "FisherContext",
    "FisherScreenPolicy",
    "FisherScreenResult",
    "FrozenCheckout",
    "GroupSequentialDesign",
    "Observable",
    "OperatingCharacteristics",
    "PairedDifference",
    "PartialEvaluator",
    "Phase",
    "PowerCurve",
    "PredictionSupport",
    "PreferencePolicy",
    "PreferenceSweep",
    "ProportionInterval",
    "RegimeSurrogate",
    "ReplicationPolicy",
    "ResultsBundle",
    "ResultsTable",
    "Scorer",
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
    "alpha_spending",
    "build_grid",
    "conditional_power",
    "coverage_curve",
    "design_group_sequential",
    "efficacy_from_share",
    "efficacy_interval",
    "efficacy_share",
    "ensemble_predict",
    "events_required",
    "expected_events_by_period",
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
    "max_events_for_power",
    "minimum_adequate",
    "operating_characteristics",
    "paired_difference",
    "paired_rank",
    "pareto_rank",
    "periods_to_events",
    "plot_calibration",
    "plot_front",
    "plot_parallel",
    "plot_scores",
    "power_curve",
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
    "unconditional_power",
    "unit_to_value",
    "value_to_unit",
    "weighted_sum_filter",
]
