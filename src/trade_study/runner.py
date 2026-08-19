"""Sweep execution: grid mode and adaptive mode.

Grid mode runs all configs via joblib parallelism.
Adaptive mode uses optuna for multi-objective Bayesian optimization.
"""

from __future__ import annotations

import inspect
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from .protocols import (
    Annotation,
    Direction,
    Observable,
    PartialEvaluator,
    ResultsTable,
    Scorer,
    Simulator,
    TrialResult,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    import optuna

    from .design import Factor

    ProgressCallback = Callable[[int, int, TrialResult], None]


def _generate_accepts_rep(world: Simulator) -> bool:
    """Whether ``world.generate`` opts into the ``rep`` convention.

    See :meth:`~trade_study.protocols.Simulator.generate` for the
    convention: a simulator that wants per-replicate stochasticity under
    ``run_grid(..., n_reps>1)`` may accept an optional keyword-only ``rep``
    parameter. Detected via introspection rather than a formal Protocol
    signature change, so simulators written before replication support
    existed keep working unmodified.

    Returns:
        True if ``world.generate``'s signature includes a ``rep`` parameter.
    """
    try:
        sig = inspect.signature(world.generate)
    except (TypeError, ValueError):
        return False
    return "rep" in sig.parameters


def _run_single(
    world: Simulator,
    scorer: Scorer,
    config: dict[str, Any],
    *,
    rep: int,
    supports_rep: bool,
) -> TrialResult:
    """Run a single trial: generate → score → return.

    Returns:
        TrialResult with config, scores, wall time, and ``rep`` metadata.
    """
    t0 = time.perf_counter()
    if supports_rep:
        # rep is an opt-in extension beyond Simulator's formal signature,
        # detected via introspection in _generate_accepts_rep.
        truth, observations = world.generate(config, rep=rep)  # type: ignore[call-arg]
    else:
        truth, observations = world.generate(config)
    scores = scorer.score(truth, observations, config)
    wall = time.perf_counter() - t0
    return TrialResult(
        config=config, scores=scores, wall_seconds=wall, metadata={"rep": rep}
    )


def run_grid(  # ruff: ignore[too-many-arguments]
    world: Simulator,
    scorer: Scorer,
    grid: list[dict[str, Any]],
    observables: list[Observable],
    *,
    annotations: list[Annotation] | None = None,
    n_jobs: int = 1,
    n_reps: int = 1,
    callback: ProgressCallback | None = None,
) -> ResultsTable:
    """Run all configurations in a grid.

    Args:
        world: Simulator that generates (truth, observations).
        scorer: Scorer that evaluates observables.
        grid: List of config dicts to evaluate.
        observables: Observable definitions (for column ordering).
        annotations: Optional external annotations (costs, etc.).
        n_jobs: Number of parallel workers (-1 for all CPUs).
        n_reps: Number of times to evaluate each design point (#112).
            When greater than 1 and ``world.generate`` accepts a ``rep``
            keyword argument, each replicate is called with its 0-indexed
            ``rep`` so the simulator can vary its own randomness; a
            simulator without a ``rep`` parameter is called unchanged,
            which reproduces its ``n_reps=1`` behavior on every replicate.
            The returned table has ``n_reps * len(grid)`` rows; each row's
            metadata carries ``design_point`` (index into ``grid``) and
            ``rep`` so replicates can be grouped, e.g. via
            :meth:`~trade_study.protocols.ResultsTable.aggregate_replicates`.
        callback: Optional progress callback invoked after each trial
            with ``(trial_index, total_trials, trial_result)``, where
            ``total_trials`` is ``n_reps * len(grid)``.

    Returns:
        ResultsTable with scored results.

    Raises:
        ValueError: If ``n_reps`` is not positive.
    """
    if n_reps < 1:
        msg = f"run_grid: n_reps must be positive, got {n_reps}"
        raise ValueError(msg)

    supports_rep = _generate_accepts_rep(world)
    tasks = [
        (design_point, cfg, rep)
        for design_point, cfg in enumerate(grid)
        for rep in range(n_reps)
    ]
    total = len(tasks)

    if n_jobs == 1:
        results: list[TrialResult] = []
        for i, (design_point, cfg, rep) in enumerate(tasks):
            r = _run_single(world, scorer, cfg, rep=rep, supports_rep=supports_rep)
            r.metadata["design_point"] = design_point
            results.append(r)
            if callback is not None:
                callback(i, total, r)
    else:
        from joblib import Parallel, delayed  # type: ignore[import-untyped]

        results = Parallel(n_jobs=n_jobs)(
            delayed(_run_single)(world, scorer, cfg, rep=rep, supports_rep=supports_rep)
            for _design_point, cfg, rep in tasks
        )
        for (design_point, _cfg, _rep), r in zip(tasks, results, strict=True):
            r.metadata["design_point"] = design_point
        if callback is not None:
            for i, r in enumerate(results):
                callback(i, total, r)

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
        metadata=[{"wall_seconds": r.wall_seconds, **r.metadata} for r in results],
    )


def run_adaptive(
    world: Simulator,
    scorer: Scorer,
    factors: list[Factor],
    observables: list[Observable],
    *,
    n_trials: int = 100,
    seed: int = 42,
) -> ResultsTable:
    """Run adaptive multi-objective optimization via optuna.

    Args:
        world: Simulator.
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
    obs_weights = [o.weight for o in observables]

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
        return tuple(
            scores.get(name, float("nan")) * w
            for name, w in zip(obs_names, obs_weights, strict=True)
        )

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


# ---------------------------------------------------------------------------
# Multi-fidelity early-stopping (#104)
# ---------------------------------------------------------------------------


def _sh_collect_observables(
    rung_results: list[list[tuple[int, float, dict[str, float]]]],
) -> list[str]:
    """Stable union of observable names across all collected rung evaluations.

    Args:
        rung_results: One list of ``(trial_idx, budget, observables)``
            tuples per rung.

    Returns:
        Sorted list of unique observable names.
    """
    names: set[str] = set()
    for rung in rung_results:
        for _, _, obs in rung:
            names.update(obs)
    return sorted(names)


def _sh_validate_inputs(
    trials: list[dict[str, Any]],
    rungs: list[float],
    eta: float,
    metric: str,
    mode: str,
) -> None:
    """Validate successive-halving arguments.

    Args:
        trials: Candidate configs.
        rungs: Budget per rung, ascending.
        eta: Halving factor.
        metric: Observable name used for ranking.
        mode: ``"min"`` or ``"max"``.

    Raises:
        ValueError: If any argument is invalid.
    """
    if not trials:
        msg = "run_successive_halving: trials must be non-empty"
        raise ValueError(msg)
    if len(rungs) < 1:
        msg = "run_successive_halving: rungs must contain at least one budget"
        raise ValueError(msg)
    if any(b <= 0 for b in rungs):
        msg = "run_successive_halving: rung budgets must be positive"
        raise ValueError(msg)
    if list(rungs) != sorted(rungs):
        msg = "run_successive_halving: rungs must be ascending"
        raise ValueError(msg)
    if eta <= 1:
        msg = "run_successive_halving: eta must be > 1"
        raise ValueError(msg)
    if not metric:
        msg = "run_successive_halving: metric must be a non-empty string"
        raise ValueError(msg)
    if mode not in {"min", "max"}:
        msg = f"run_successive_halving: mode must be 'min' or 'max', got {mode!r}"
        raise ValueError(msg)


def run_successive_halving(
    trials: list[dict[str, Any]],
    sim: PartialEvaluator,
    *,
    rungs: list[float],
    eta: float = 3.0,
    metric: str,
    mode: str = "min",
) -> ResultsTable:
    """Successive-halving multi-fidelity early-stopping (#104).

    Evaluates every trial at the lowest rung, keeps the top ``1/eta`` by
    ``metric`` (according to ``mode``), promotes survivors to the next
    budget, and repeats until the highest rung. Every (trial, rung)
    evaluation is recorded as one row in the returned :class:`ResultsTable`,
    with ``rung`` index and ``budget`` stored in per-row metadata.

    Args:
        trials: Candidate configurations to evaluate.
        sim: A :class:`PartialEvaluator` whose ``evaluate(config, budget)``
            returns observables including ``metric``.
        rungs: Strictly ascending list of budgets (e.g. epochs, iterations).
            Length determines the number of halving rounds.
        eta: Reduction factor between rungs (>1). Each rung keeps
            ``ceil(n_prev / eta)`` survivors. Defaults to 3 per Li et al.
            (2017).
        metric: Observable name used to rank trials at each rung.
        mode: ``"min"`` (lower is better) or ``"max"``.

    Returns:
        :class:`ResultsTable` whose rows are (trial, rung) evaluations.
        Per-row metadata contains ``rung`` (0-indexed), ``budget``,
        ``trial_index`` (position in the input ``trials`` list),
        ``promoted`` (whether this trial advanced past this rung), and
        ``wall_seconds``. Propagates :class:`ValueError` from the input
        validator when arguments are invalid.

    Raises:
        KeyError: If ``metric`` is missing from a returned observables
            dict.
    """
    _sh_validate_inputs(trials, rungs, eta, metric, mode)

    # rung_records[r] = list of (trial_idx, budget, observables) at rung r
    rung_records: list[list[tuple[int, float, dict[str, float], float]]] = [
        [] for _ in rungs
    ]
    survivors: list[int] = list(range(len(trials)))

    for r, budget in enumerate(rungs):
        for trial_idx in survivors:
            t0 = time.perf_counter()
            obs = sim.evaluate(trials[trial_idx], budget)
            wall = time.perf_counter() - t0
            if metric not in obs:
                msg = (
                    f"run_successive_halving: PartialEvaluator did not return "
                    f"metric {metric!r} at rung {r} for trial {trial_idx}"
                )
                raise KeyError(msg)
            rung_records[r].append((trial_idx, budget, obs, wall))

        if r < len(rungs) - 1:
            ranked = sorted(
                rung_records[r],
                key=lambda row: row[2][metric],
                reverse=(mode == "max"),
            )
            n_keep = max(1, int(np.ceil(len(ranked) / eta)))
            survivors = [row[0] for row in ranked[:n_keep]]

    obs_names = _sh_collect_observables([
        [(idx, b, o) for idx, b, o, _w in rung] for rung in rung_records
    ])

    promoted_at_rung: list[set[int]] = [set() for _ in rungs]
    for r in range(len(rungs) - 1):
        ranked = sorted(
            rung_records[r],
            key=lambda row: row[2][metric],
            reverse=(mode == "max"),
        )
        n_keep = max(1, int(np.ceil(len(ranked) / eta)))
        promoted_at_rung[r] = {row[0] for row in ranked[:n_keep]}

    configs: list[dict[str, Any]] = []
    score_rows: list[list[float]] = []
    metadata: list[dict[str, Any]] = []
    for r, rung in enumerate(rung_records):
        for trial_idx, budget, obs, wall in rung:
            configs.append(trials[trial_idx])
            score_rows.append([obs.get(name, float("nan")) for name in obs_names])
            metadata.append({
                "rung": r,
                "budget": budget,
                "trial_index": trial_idx,
                "promoted": trial_idx in promoted_at_rung[r],
                "wall_seconds": wall,
            })

    return ResultsTable(
        configs=configs,
        scores=np.array(score_rows) if score_rows else np.zeros((0, len(obs_names))),
        observable_names=obs_names,
        metadata=metadata,
    )


def _hyperband_brackets(
    max_budget: float,
    eta: float,
) -> list[tuple[int, float]]:
    """Compute the (n_trials, min_budget) per Hyperband bracket.

    Implements the bracket schedule from Li et al. (2017), Algorithm 1.

    Args:
        max_budget: Maximum resource ``R`` allocated to a single trial.
        eta: Reduction factor (>1).

    Returns:
        List of ``(n_initial_trials, min_budget)`` tuples, one per bracket.
    """
    s_max = int(np.floor(np.log(max_budget) / np.log(eta)))
    budget_total = (s_max + 1) * max_budget
    brackets: list[tuple[int, float]] = []
    for s in range(s_max, -1, -1):
        n = int(np.ceil(budget_total / max_budget * eta**s / (s + 1)))
        r = max_budget * eta ** (-s)
        brackets.append((n, r))
    return brackets


def run_hyperband(
    trial_factory: Callable[[int, int], list[dict[str, Any]]],
    sim: PartialEvaluator,
    *,
    max_budget: float,
    eta: float = 3.0,
    metric: str,
    mode: str = "min",
) -> ResultsTable:
    """Hyperband: multi-bracket successive-halving (#104).

    Wraps :func:`run_successive_halving` with the bracket schedule from
    Li et al. (2017). Each bracket trades off the number of initial
    trials against the minimum budget per trial; together they hedge
    against picking either ratio wrong.

    Args:
        trial_factory: Callable ``(bracket_index, n_trials) -> trials`` that
            returns a fresh list of candidate configs for each bracket.
            Typically this wraps :func:`trade_study.build_grid` with a
            bracket-derived seed so brackets sample different points.
        sim: A :class:`PartialEvaluator`.
        max_budget: Maximum resource ``R`` per trial.
        eta: Reduction factor (>1). Defaults to 3.
        metric: Observable used for ranking within each bracket.
        mode: ``"min"`` or ``"max"``.

    Returns:
        Concatenated :class:`ResultsTable` across all brackets, with an
        additional ``bracket`` field in each row's metadata.

    Raises:
        ValueError: If ``max_budget <= 0`` or ``eta <= 1``.
    """
    if max_budget <= 0:
        msg = "run_hyperband: max_budget must be positive"
        raise ValueError(msg)
    if eta <= 1:
        msg = "run_hyperband: eta must be > 1"
        raise ValueError(msg)

    brackets = _hyperband_brackets(max_budget, eta)

    all_configs: list[dict[str, Any]] = []
    all_scores: list[list[float]] = []
    all_metadata: list[dict[str, Any]] = []
    obs_names: list[str] = []

    for bracket_idx, (n_initial, r_min) in enumerate(brackets):
        trials = trial_factory(bracket_idx, n_initial)
        if not trials:
            continue
        s = len(brackets) - 1 - bracket_idx
        rungs = [r_min * eta**i for i in range(s + 1)]
        bracket_results = run_successive_halving(
            trials,
            sim,
            rungs=rungs,
            eta=eta,
            metric=metric,
            mode=mode,
        )
        if not obs_names:
            obs_names = bracket_results.observable_names
        elif obs_names != bracket_results.observable_names:
            # Pad / reorder so all brackets share the column layout.
            union = sorted(set(obs_names) | set(bracket_results.observable_names))
            obs_names = union

        for i, cfg in enumerate(bracket_results.configs):
            all_configs.append(cfg)
            row = bracket_results.scores[i]
            all_scores.append([
                float(row[bracket_results.observable_names.index(n)])
                if n in bracket_results.observable_names
                else float("nan")
                for n in obs_names
            ])
            meta = dict(bracket_results.metadata[i])
            meta["bracket"] = bracket_idx
            all_metadata.append(meta)

    return ResultsTable(
        configs=all_configs,
        scores=np.array(all_scores) if all_scores else np.zeros((0, len(obs_names))),
        observable_names=obs_names,
        metadata=all_metadata,
    )
