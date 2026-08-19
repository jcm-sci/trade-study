"""Tests for runner module (issues #18, #19, #20)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from trade_study.design import Factor, FactorType
from trade_study.protocols import (
    Annotation,
    Direction,
    Observable,
    ResultsTable,
    TrialResult,
)
from trade_study.runner import (
    run_adaptive,
    run_grid,
    run_hyperband,
    run_successive_halving,
)

# ---------------------------------------------------------------------------
# Toy implementations
# ---------------------------------------------------------------------------


class _ToySimulator:
    """Simulator that returns config values as truth and observations."""

    def generate(self, config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        """Pass config through as both truth and observations.

        Returns:
            Tuple of (config, config).
        """
        return config, config


class _ToyScorer:
    """Scorer that computes simple metrics from config values."""

    def score(
        self,
        truth: Any,
        observations: Any,
        config: dict[str, Any],
    ) -> dict[str, float]:
        """Score: error = |alpha - 0.5|, cost = alpha * 10.

        Returns:
            Dict with ``error`` and ``cost`` scores.
        """
        a = float(config.get("alpha", 0.5))
        return {"error": abs(a - 0.5), "cost": a * 10.0}


@pytest.fixture
def world() -> _ToySimulator:
    """Toy simulator fixture.

    Returns:
        A _ToySimulator instance.
    """
    return _ToySimulator()


@pytest.fixture
def scorer() -> _ToyScorer:
    """Toy scorer fixture.

    Returns:
        A _ToyScorer instance.
    """
    return _ToyScorer()


@pytest.fixture
def observables() -> list[Observable]:
    """Two observables: error (minimize) and cost (minimize).

    Returns:
        List of two Observable instances.
    """
    return [
        Observable("error", Direction.MINIMIZE),
        Observable("cost", Direction.MINIMIZE),
    ]


@pytest.fixture
def grid() -> list[dict[str, Any]]:
    """Simple 5-point grid over alpha.

    Returns:
        List of config dicts.
    """
    return [{"alpha": v} for v in [0.0, 0.25, 0.5, 0.75, 1.0]]


# ---------------------------------------------------------------------------
# run_grid — serial (#18)
# ---------------------------------------------------------------------------


def test_run_grid_serial_returns_results_table(
    world: _ToySimulator,
    scorer: _ToyScorer,
    grid: list[dict[str, Any]],
    observables: list[Observable],
) -> None:
    result = run_grid(world, scorer, grid, observables)
    assert result.scores.shape == (5, 2)


def test_run_grid_serial_all_configs_evaluated(
    world: _ToySimulator,
    scorer: _ToyScorer,
    grid: list[dict[str, Any]],
    observables: list[Observable],
) -> None:
    result = run_grid(world, scorer, grid, observables)
    assert len(result.configs) == 5


def test_run_grid_serial_observable_names(
    world: _ToySimulator,
    scorer: _ToyScorer,
    grid: list[dict[str, Any]],
    observables: list[Observable],
) -> None:
    result = run_grid(world, scorer, grid, observables)
    assert result.observable_names == ["error", "cost"]


def test_run_grid_serial_score_values(
    world: _ToySimulator,
    scorer: _ToyScorer,
    grid: list[dict[str, Any]],
    observables: list[Observable],
) -> None:
    result = run_grid(world, scorer, grid, observables)
    # alpha=0.5 → error=0.0
    idx = 2  # alpha=0.5
    assert result.scores[idx, 0] == pytest.approx(0.0)
    # alpha=0.0 → cost=0.0
    assert result.scores[0, 1] == pytest.approx(0.0)


def test_run_grid_serial_metadata_has_wall_seconds(
    world: _ToySimulator,
    scorer: _ToyScorer,
    grid: list[dict[str, Any]],
    observables: list[Observable],
) -> None:
    result = run_grid(world, scorer, grid, observables)
    assert all("wall_seconds" in m for m in result.metadata)
    assert all(m["wall_seconds"] >= 0.0 for m in result.metadata)


def test_run_grid_serial_with_annotations(
    world: _ToySimulator,
    scorer: _ToyScorer,
    observables: list[Observable],
) -> None:
    grid = [{"alpha": 0.2, "method": "a"}, {"alpha": 0.8, "method": "b"}]
    annotations = [
        Annotation(name="method_cost", lookup={"a": 10.0, "b": 20.0}, key="method"),
    ]
    result = run_grid(world, scorer, grid, observables, annotations=annotations)
    assert result.annotations is not None
    assert result.annotations.shape == (2, 1)
    assert result.annotation_names == ["method_cost"]
    assert result.annotations[0, 0] == pytest.approx(10.0)
    assert result.annotations[1, 0] == pytest.approx(20.0)


def test_run_grid_serial_no_annotations(
    world: _ToySimulator,
    scorer: _ToyScorer,
    grid: list[dict[str, Any]],
    observables: list[Observable],
) -> None:
    result = run_grid(world, scorer, grid, observables)
    assert result.annotations is None
    assert result.annotation_names == []


# ---------------------------------------------------------------------------
# run_grid — parallel (#19)
# ---------------------------------------------------------------------------


def test_run_grid_parallel_same_results(
    world: _ToySimulator,
    scorer: _ToyScorer,
    grid: list[dict[str, Any]],
    observables: list[Observable],
) -> None:
    serial = run_grid(world, scorer, grid, observables, n_jobs=1)
    parallel = run_grid(world, scorer, grid, observables, n_jobs=2)
    np.testing.assert_allclose(serial.scores, parallel.scores)


def test_run_grid_parallel_same_configs(
    world: _ToySimulator,
    scorer: _ToyScorer,
    grid: list[dict[str, Any]],
    observables: list[Observable],
) -> None:
    serial = run_grid(world, scorer, grid, observables, n_jobs=1)
    parallel = run_grid(world, scorer, grid, observables, n_jobs=2)
    assert serial.configs == parallel.configs


def test_run_grid_parallel_shape(
    world: _ToySimulator,
    scorer: _ToyScorer,
    grid: list[dict[str, Any]],
    observables: list[Observable],
) -> None:
    result = run_grid(world, scorer, grid, observables, n_jobs=2)
    assert result.scores.shape == (5, 2)


def test_run_grid_parallel_metadata(
    world: _ToySimulator,
    scorer: _ToyScorer,
    grid: list[dict[str, Any]],
    observables: list[Observable],
) -> None:
    result = run_grid(world, scorer, grid, observables, n_jobs=2)
    assert all("wall_seconds" in m for m in result.metadata)


# ---------------------------------------------------------------------------
# run_adaptive (#20)
# ---------------------------------------------------------------------------


def test_run_adaptive_returns_n_trials(
    world: _ToySimulator,
    scorer: _ToyScorer,
    observables: list[Observable],
) -> None:
    factors = [Factor("alpha", FactorType.CONTINUOUS, bounds=(0.0, 1.0))]
    result = run_adaptive(world, scorer, factors, observables, n_trials=20)
    assert len(result.configs) == 20
    assert result.scores.shape == (20, 2)


def test_run_adaptive_observable_names(
    world: _ToySimulator,
    scorer: _ToyScorer,
    observables: list[Observable],
) -> None:
    factors = [Factor("alpha", FactorType.CONTINUOUS, bounds=(0.0, 1.0))]
    result = run_adaptive(world, scorer, factors, observables, n_trials=10)
    assert result.observable_names == ["error", "cost"]


def test_run_adaptive_minimize_direction(
    world: _ToySimulator,
    scorer: _ToyScorer,
) -> None:
    observables = [Observable("error", Direction.MINIMIZE)]
    factors = [Factor("alpha", FactorType.CONTINUOUS, bounds=(0.0, 1.0))]
    result = run_adaptive(
        world,
        scorer,
        factors,
        observables,
        n_trials=30,
        seed=42,
    )
    # Best error should be near 0 (alpha near 0.5)
    assert np.min(result.scores[:, 0]) < 0.1


def test_run_adaptive_maximize_direction() -> None:
    class _MaxScorer:
        def score(
            self,
            truth: Any,
            observations: Any,
            config: dict[str, Any],
        ) -> dict[str, float]:
            """Maximize: quality = -(alpha - 0.8)^2.

            Returns:
                Dict with ``quality`` score.
            """
            a = float(config.get("alpha", 0.0))
            return {"quality": -((a - 0.8) ** 2)}

    observables = [Observable("quality", Direction.MAXIMIZE)]
    factors = [Factor("alpha", FactorType.CONTINUOUS, bounds=(0.0, 1.0))]
    result = run_adaptive(
        _ToySimulator(),
        _MaxScorer(),
        factors,
        observables,
        n_trials=30,
    )
    assert result.scores.shape[0] == 30


def test_run_adaptive_categorical_factor(
    world: _ToySimulator,
    scorer: _ToyScorer,
    observables: list[Observable],
) -> None:
    factors = [
        Factor("alpha", FactorType.CONTINUOUS, bounds=(0.0, 1.0)),
        Factor("method", FactorType.CATEGORICAL, levels=["a", "b"]),
    ]
    result = run_adaptive(world, scorer, factors, observables, n_trials=15)
    assert len(result.configs) == 15
    methods = {cfg["method"] for cfg in result.configs}
    assert methods <= {"a", "b"}


def test_run_adaptive_deterministic_seed(
    world: _ToySimulator,
    scorer: _ToyScorer,
    observables: list[Observable],
) -> None:
    factors = [Factor("alpha", FactorType.CONTINUOUS, bounds=(0.0, 1.0))]
    r1 = run_adaptive(world, scorer, factors, observables, n_trials=10, seed=7)
    r2 = run_adaptive(world, scorer, factors, observables, n_trials=10, seed=7)
    np.testing.assert_allclose(r1.scores, r2.scores)


# ---------------------------------------------------------------------------
# Progress callback (#77)
# ---------------------------------------------------------------------------


def test_run_grid_callback_called(
    world: _ToySimulator,
    scorer: _ToyScorer,
    observables: list[Observable],
) -> None:
    """Callback is invoked once per trial with correct arguments."""
    grid = [{"alpha": v} for v in [0.0, 0.25, 0.5]]
    calls: list[tuple[int, int, TrialResult]] = []
    run_grid(
        world,
        scorer,
        grid,
        observables,
        callback=lambda i, n, r: calls.append((i, n, r)),
    )
    assert len(calls) == 3
    for i, (idx, total, result) in enumerate(calls):
        assert idx == i
        assert total == 3
        assert isinstance(result, TrialResult)


def test_run_grid_callback_none(
    world: _ToySimulator,
    scorer: _ToyScorer,
    observables: list[Observable],
) -> None:
    """No callback (default) runs without error."""
    grid = [{"alpha": 0.5}]
    result = run_grid(world, scorer, grid, observables)
    assert len(result.configs) == 1


# ---------------------------------------------------------------------------
# Replicated trials (#112)
# ---------------------------------------------------------------------------


class _RepAwareSimulator:
    """Simulator whose observations vary deterministically with ``rep``."""

    def generate(
        self, config: dict[str, Any], *, rep: int = 0
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Offset alpha by rep so replicate variation is observable.

        Returns:
            Tuple of (varied config, varied config) used as truth/observations.
        """
        varied = {**config, "alpha": float(config.get("alpha", 0.5)) + 0.01 * rep}
        return varied, varied


class _RepSensitiveScorer:
    """Scorer that reads alpha from observations, not the original config."""

    def score(
        self,
        truth: Any,
        observations: Any,
        config: dict[str, Any],
    ) -> dict[str, float]:
        """Score from observations so rep-driven variation is visible.

        Returns:
            Dict with ``error`` and ``cost`` scores derived from observations.
        """
        a = float(observations.get("alpha", 0.5))
        return {"error": abs(a - 0.5), "cost": a * 10.0}


@pytest.fixture
def rep_world() -> _RepAwareSimulator:
    """Rep-aware simulator fixture.

    Returns:
        A _RepAwareSimulator instance.
    """
    return _RepAwareSimulator()


@pytest.fixture
def rep_scorer() -> _RepSensitiveScorer:
    """Rep-sensitive scorer fixture.

    Returns:
        A _RepSensitiveScorer instance.
    """
    return _RepSensitiveScorer()


def test_run_grid_n_reps_default_matches_n_reps_one(
    world: _ToySimulator,
    scorer: _ToyScorer,
    grid: list[dict[str, Any]],
    observables: list[Observable],
) -> None:
    """n_reps defaults to 1: row count and scores unchanged from before #112."""
    default = run_grid(world, scorer, grid, observables)
    explicit = run_grid(world, scorer, grid, observables, n_reps=1)
    assert len(default.configs) == len(grid)
    np.testing.assert_allclose(default.scores, explicit.scores)


def test_run_grid_n_reps_rejects_non_positive(
    world: _ToySimulator,
    scorer: _ToyScorer,
    grid: list[dict[str, Any]],
    observables: list[Observable],
) -> None:
    with pytest.raises(ValueError, match="n_reps must be positive"):
        run_grid(world, scorer, grid, observables, n_reps=0)


def test_run_grid_n_reps_expands_row_count(
    world: _ToySimulator,
    scorer: _ToyScorer,
    grid: list[dict[str, Any]],
    observables: list[Observable],
) -> None:
    result = run_grid(world, scorer, grid, observables, n_reps=3)
    assert len(result.configs) == 3 * len(grid)
    assert result.scores.shape == (3 * len(grid), 2)


def test_run_grid_n_reps_metadata_design_point_and_rep(
    world: _ToySimulator,
    scorer: _ToyScorer,
    grid: list[dict[str, Any]],
    observables: list[Observable],
) -> None:
    result = run_grid(world, scorer, grid, observables, n_reps=3)
    design_points = [m["design_point"] for m in result.metadata]
    reps = [m["rep"] for m in result.metadata]
    assert design_points == [dp for dp in range(len(grid)) for _ in range(3)]
    assert reps == [r for _dp in range(len(grid)) for r in range(3)]


def test_run_grid_n_reps_non_rep_aware_simulator_repeats_identically(
    world: _ToySimulator,
    scorer: _ToyScorer,
    observables: list[Observable],
) -> None:
    """A simulator without a rep parameter is called unchanged each replicate."""
    grid = [{"alpha": 0.3}]
    result = run_grid(world, scorer, grid, observables, n_reps=4)
    np.testing.assert_allclose(result.scores[0], result.scores[1])
    np.testing.assert_allclose(result.scores[0], result.scores[3])


def test_run_grid_n_reps_rep_aware_simulator_varies_by_rep(
    rep_world: _RepAwareSimulator,
    rep_scorer: _RepSensitiveScorer,
    observables: list[Observable],
) -> None:
    grid = [{"alpha": 0.3}]
    result = run_grid(rep_world, rep_scorer, grid, observables, n_reps=3)
    assert not np.allclose(result.scores[0], result.scores[1])
    assert not np.allclose(result.scores[1], result.scores[2])


def test_run_grid_n_reps_parallel_matches_serial(
    rep_world: _RepAwareSimulator,
    rep_scorer: _RepSensitiveScorer,
    grid: list[dict[str, Any]],
    observables: list[Observable],
) -> None:
    serial = run_grid(rep_world, rep_scorer, grid, observables, n_reps=2, n_jobs=1)
    parallel = run_grid(rep_world, rep_scorer, grid, observables, n_reps=2, n_jobs=2)
    np.testing.assert_allclose(serial.scores, parallel.scores)
    serial_dp = [m["design_point"] for m in serial.metadata]
    parallel_dp = [m["design_point"] for m in parallel.metadata]
    assert serial_dp == parallel_dp


def test_run_grid_n_reps_callback_total_reflects_replicates(
    rep_world: _RepAwareSimulator,
    rep_scorer: _RepSensitiveScorer,
    observables: list[Observable],
) -> None:
    grid = [{"alpha": 0.0}, {"alpha": 1.0}]
    calls: list[tuple[int, int]] = []
    run_grid(
        rep_world,
        rep_scorer,
        grid,
        observables,
        n_reps=3,
        callback=lambda i, n, _r: calls.append((i, n)),
    )
    assert len(calls) == 6
    assert all(total == 6 for _idx, total in calls)


def test_aggregate_replicates_groups_by_design_point(
    rep_world: _RepAwareSimulator,
    rep_scorer: _RepSensitiveScorer,
    grid: list[dict[str, Any]],
    observables: list[Observable],
) -> None:
    raw = run_grid(rep_world, rep_scorer, grid, observables, n_reps=4)
    agg = raw.aggregate_replicates()
    assert len(agg.configs) == len(grid)
    assert agg.observable_names == raw.observable_names
    for i in range(len(grid)):
        rows = [j for j, m in enumerate(raw.metadata) if m["design_point"] == i]
        np.testing.assert_allclose(agg.scores[i], raw.scores[rows].mean(axis=0))


def test_aggregate_replicates_records_n_reps_and_std(
    rep_world: _RepAwareSimulator,
    rep_scorer: _RepSensitiveScorer,
    grid: list[dict[str, Any]],
    observables: list[Observable],
) -> None:
    raw = run_grid(rep_world, rep_scorer, grid, observables, n_reps=4)
    agg = raw.aggregate_replicates()
    assert all(m["n_reps"] == 4 for m in agg.metadata)
    assert all("score_std" in m for m in agg.metadata)
    assert all(name in agg.metadata[0]["score_std"] for name in raw.observable_names)


def test_aggregate_replicates_missing_design_point_raises(
    observables: list[Observable],
) -> None:
    table = ResultsTable(
        configs=[{"alpha": 0.5}],
        scores=np.array([[0.0, 5.0]]),
        observable_names=[o.name for o in observables],
        metadata=[{"wall_seconds": 0.0}],
    )
    with pytest.raises(KeyError, match="design_point"):
        table.aggregate_replicates()


# ---------------------------------------------------------------------------
# Successive halving / Hyperband (#104)
# ---------------------------------------------------------------------------


class _ToyPartialEvaluator:
    """Toy PartialEvaluator: loss decays as `target * exp(-budget/scale)` + noise.

    Trials with smaller `target` reach lower loss faster, so they should
    survive successive halving.
    """

    def evaluate(
        self,
        config: dict[str, Any],
        budget: float,
    ) -> dict[str, float]:
        """Return a budget-decayed loss for ``config``.

        Args:
            config: Must contain ``target`` (asymptotic loss).
            budget: Resource budget; larger ⇒ better fidelity.

        Returns:
            Dict with ``loss`` and ``budget`` observables.
        """
        target = float(config["target"])
        loss = target + np.exp(-budget / 5.0)
        return {"loss": loss, "budget": float(budget)}


def test_successive_halving_keeps_best() -> None:
    sim = _ToyPartialEvaluator()
    trials = [{"target": t} for t in [0.1, 0.5, 0.9, 0.05, 0.7, 0.3, 0.6, 0.2, 0.8]]
    results = run_successive_halving(
        trials,
        sim,
        rungs=[1.0, 3.0, 9.0],
        eta=3.0,
        metric="loss",
        mode="min",
    )
    # Final rung should contain ceil(ceil(9/3)/3) = 1 trial,
    # which must be the lowest target.
    final = [m for m in results.metadata if m["rung"] == 2]
    assert len(final) == 1
    survivor_idx = final[0]["trial_index"]
    assert trials[survivor_idx]["target"] == pytest.approx(0.05)


def test_successive_halving_row_count() -> None:
    sim = _ToyPartialEvaluator()
    trials = [{"target": t / 10} for t in range(9)]
    results = run_successive_halving(
        trials,
        sim,
        rungs=[1.0, 3.0, 9.0],
        eta=3.0,
        metric="loss",
        mode="min",
    )
    # Rung sizes: 9, 3, 1 → 13 rows.
    assert len(results.configs) == 9 + 3 + 1
    assert results.scores.shape == (13, 2)


def test_successive_halving_records_metadata() -> None:
    sim = _ToyPartialEvaluator()
    trials = [{"target": 0.1}, {"target": 0.9}]
    results = run_successive_halving(
        trials,
        sim,
        rungs=[1.0, 3.0],
        eta=2.0,
        metric="loss",
        mode="min",
    )
    keys = set(results.metadata[0])
    assert {"rung", "budget", "trial_index", "promoted", "wall_seconds"} <= keys
    promoted_first_rung = [
        m for m in results.metadata if m["rung"] == 0 and m["promoted"]
    ]
    assert len(promoted_first_rung) == 1


def test_successive_halving_max_mode() -> None:
    sim = _ToyPartialEvaluator()
    trials = [{"target": t} for t in [0.1, 0.9]]
    # In max mode, the higher loss wins.
    results = run_successive_halving(
        trials,
        sim,
        rungs=[1.0, 3.0],
        eta=2.0,
        metric="loss",
        mode="max",
    )
    final = [m for m in results.metadata if m["rung"] == 1]
    assert len(final) == 1
    assert trials[final[0]["trial_index"]]["target"] == pytest.approx(0.9)


def test_successive_halving_validation() -> None:
    sim = _ToyPartialEvaluator()
    trials = [{"target": 0.1}]
    with pytest.raises(ValueError, match="trials must be non-empty"):
        run_successive_halving([], sim, rungs=[1.0], metric="loss")
    with pytest.raises(ValueError, match="ascending"):
        run_successive_halving(trials, sim, rungs=[3.0, 1.0], metric="loss")
    with pytest.raises(ValueError, match="positive"):
        run_successive_halving(trials, sim, rungs=[0.0], metric="loss")
    with pytest.raises(ValueError, match="eta must be > 1"):
        run_successive_halving(trials, sim, rungs=[1.0], eta=1.0, metric="loss")
    with pytest.raises(ValueError, match="metric must be"):
        run_successive_halving(trials, sim, rungs=[1.0], metric="")
    with pytest.raises(ValueError, match="mode must be"):
        run_successive_halving(trials, sim, rungs=[1.0], metric="loss", mode="bogus")
    with pytest.raises(ValueError, match="rungs must contain"):
        run_successive_halving(trials, sim, rungs=[], metric="loss")


def test_successive_halving_missing_metric() -> None:
    class _NoMetric:
        def evaluate(self, _config: dict[str, Any], _budget: float) -> dict[str, float]:
            return {"other": 1.0}

    with pytest.raises(KeyError, match="did not return metric"):
        run_successive_halving(
            [{"target": 0.1}],
            _NoMetric(),
            rungs=[1.0],
            metric="loss",
        )


def test_hyperband_runs_all_brackets() -> None:
    sim = _ToyPartialEvaluator()
    rng = np.random.default_rng(0)

    def factory(bracket_idx: int, n: int) -> list[dict[str, Any]]:
        # Bracket-seeded random targets so brackets explore different points.
        local_rng = np.random.default_rng(bracket_idx + 1)
        return [{"target": float(local_rng.uniform(0.0, 1.0))} for _ in range(n)]

    _ = rng  # silence unused
    results = run_hyperband(
        factory,
        sim,
        max_budget=9.0,
        eta=3.0,
        metric="loss",
        mode="min",
    )
    brackets_seen = {m["bracket"] for m in results.metadata}
    # eta=3, R=9 → s_max = 2 → 3 brackets (s = 2, 1, 0).
    assert brackets_seen == {0, 1, 2}
    assert "loss" in results.observable_names


def test_hyperband_validation() -> None:
    sim = _ToyPartialEvaluator()

    def factory(_b: int, _n: int) -> list[dict[str, Any]]:
        return [{"target": 0.1}]

    with pytest.raises(ValueError, match="max_budget must be positive"):
        run_hyperband(factory, sim, max_budget=0.0, metric="loss")
    with pytest.raises(ValueError, match="eta must be > 1"):
        run_hyperband(factory, sim, max_budget=9.0, eta=1.0, metric="loss")


def test_partial_evaluator_protocol_runtime_check() -> None:
    from trade_study.protocols import PartialEvaluator

    assert isinstance(_ToyPartialEvaluator(), PartialEvaluator)
