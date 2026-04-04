"""Sweep execution: grid mode and adaptive mode.

Grid mode runs all configs via joblib parallelism.
Adaptive mode uses optuna for multi-objective Bayesian optimization.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np

from .protocols import (
    Annotation,
    Direction,
    ModelWorld,
    Observable,
    ResultsTable,
    Scorer,
    TrialResult,
)

if TYPE_CHECKING:
    import optuna

    from .design import Factor


def _run_single(
    world: ModelWorld,
    scorer: Scorer,
    config: dict[str, Any],
) -> TrialResult:
    """Run a single trial: generate → score → return.

    Returns:
        TrialResult with config, scores, and wall time.
    """
    t0 = time.perf_counter()
    truth, observations = world.generate(config)
    scores = scorer.score(truth, observations, config)
    wall = time.perf_counter() - t0
    return TrialResult(config=config, scores=scores, wall_seconds=wall)


def run_grid(
    world: ModelWorld,
    scorer: Scorer,
    grid: list[dict[str, Any]],
    observables: list[Observable],
    *,
    annotations: list[Annotation] | None = None,
    n_jobs: int = 1,
) -> ResultsTable:
    """Run all configurations in a grid.

    Args:
        world: Model world that generates (truth, observations).
        scorer: Scorer that evaluates observables.
        grid: List of config dicts to evaluate.
        observables: Observable definitions (for column ordering).
        annotations: Optional external annotations (costs, etc.).
        n_jobs: Number of parallel workers (-1 for all CPUs).

    Returns:
        ResultsTable with scored results.
    """
    if n_jobs == 1:
        results = [_run_single(world, scorer, cfg) for cfg in grid]
    else:
        from joblib import Parallel, delayed  # type: ignore[import-untyped]

        results = Parallel(n_jobs=n_jobs)(
            delayed(_run_single)(world, scorer, cfg) for cfg in grid
        )

    obs_names = [o.name for o in observables]
    score_matrix = np.array([
        [r.scores.get(name, np.nan) for name in obs_names] for r in results
    ])

    ann_matrix = None
    ann_names: list[str] = []
    if annotations:
        ann_names = [a.name for a in annotations]
        ann_matrix = np.array([
            [a.resolve(r.config) for a in annotations] for r in results
        ])

    return ResultsTable(
        configs=[r.config for r in results],
        scores=score_matrix,
        observable_names=obs_names,
        annotations=ann_matrix,
        annotation_names=ann_names,
        metadata=[{"wall_seconds": r.wall_seconds} for r in results],
    )


def run_adaptive(
    world: ModelWorld,
    scorer: Scorer,
    factors: list[Factor],
    observables: list[Observable],
    *,
    n_trials: int = 100,
    seed: int = 42,
) -> ResultsTable:
    """Run adaptive multi-objective optimization via optuna.

    Args:
        world: Model world.
        scorer: Scorer for observables.
        factors: Factor definitions (from design module).
        observables: Observable definitions.
        n_trials: Number of optuna trials.
        seed: Random seed.

    Returns:
        ResultsTable with scored results.
    """
    import optuna as _optuna

    from .design import FactorType

    directions_str = [
        "minimize" if o.direction == Direction.MINIMIZE else "maximize"
        for o in observables
    ]

    study = _optuna.create_study(
        directions=directions_str,
        sampler=_optuna.samplers.NSGAIISampler(seed=seed),
    )

    obs_names = [o.name for o in observables]

    def objective(trial: optuna.trial.Trial) -> tuple[float, ...]:
        config: dict[str, Any] = {}
        for f in factors:
            if f.factor_type == FactorType.CONTINUOUS and f.bounds is not None:
                config[f.name] = trial.suggest_float(
                    f.name,
                    f.bounds[0],
                    f.bounds[1],
                )
            elif f.levels is not None and f.factor_type in {
                FactorType.CATEGORICAL,
                FactorType.DISCRETE,
            }:
                config[f.name] = trial.suggest_categorical(f.name, f.levels)
        truth, observations = world.generate(config)
        scores = scorer.score(truth, observations, config)
        return tuple(scores.get(name, float("nan")) for name in obs_names)

    _optuna.logging.set_verbosity(_optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials)

    configs = []
    score_rows = []
    for trial in study.trials:
        configs.append(trial.params)
        score_rows.append(list(trial.values))

    return ResultsTable(
        configs=configs,
        scores=np.array(score_rows),
        observable_names=obs_names,
    )
