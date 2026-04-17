"""Tests for study module (issues #21, #22, #23, #25)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from trade_study.design import Factor, FactorType
from trade_study.protocols import (
    Annotation,
    Constraint,
    Direction,
    Observable,
    ResultsTable,
    TrialResult,
)
from trade_study.study import (
    Phase,
    Study,
    feasibility_filter,
    top_k_pareto_filter,
    weighted_sum_filter,
)

# ---------------------------------------------------------------------------
# Toy implementations (same pattern as test_runner)
# ---------------------------------------------------------------------------


class _ToySimulator:
    """Simulator that passes config through as truth and observations."""

    def generate(
        self,
        config: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return config as both truth and observations.

        Returns:
            Tuple of (config, config).
        """
        return config, config


class _ToyScorer:
    """Scorer that computes error and cost from config values."""

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
# top_k_pareto_filter (#22)
# ---------------------------------------------------------------------------


def test_top_k_pareto_filter_returns_callable() -> None:
    fn = top_k_pareto_filter(k=3)
    assert callable(fn)


def test_top_k_pareto_filter_keeps_at_most_k(
    observables: list[Observable],
) -> None:
    rt = ResultsTable(
        configs=[{"alpha": v} for v in [0.0, 0.25, 0.5, 0.75, 1.0]],
        scores=np.array([
            [0.5, 0.0],
            [0.25, 2.5],
            [0.0, 5.0],
            [0.25, 7.5],
            [0.5, 10.0],
        ]),
        observable_names=["error", "cost"],
    )
    fn = top_k_pareto_filter(k=3)
    indices = fn(rt, observables)
    assert len(indices) <= 3


def test_top_k_pareto_filter_returns_best_ranks(
    observables: list[Observable],
) -> None:
    # Pareto front: row 0 (error=0.5, cost=0), row 2 (error=0, cost=5)
    # are non-dominated for minimize/minimize
    rt = ResultsTable(
        configs=[{"alpha": v} for v in [0.0, 0.25, 0.5, 0.75, 1.0]],
        scores=np.array([
            [0.5, 0.0],
            [0.25, 2.5],
            [0.0, 5.0],
            [0.25, 7.5],
            [0.5, 10.0],
        ]),
        observable_names=["error", "cost"],
    )
    fn = top_k_pareto_filter(k=2)
    indices = fn(rt, observables)
    # rank-1 configs should be included
    assert 0 in indices or 2 in indices


def test_top_k_pareto_filter_with_objective_subset() -> None:
    observables = [
        Observable("error", Direction.MINIMIZE),
        Observable("cost", Direction.MINIMIZE),
    ]
    rt = ResultsTable(
        configs=[{"a": 1}, {"a": 2}, {"a": 3}],
        scores=np.array([[1.0, 10.0], [2.0, 5.0], [3.0, 1.0]]),
        observable_names=["error", "cost"],
    )
    fn = top_k_pareto_filter(k=2, objective_names=["error"])
    indices = fn(rt, observables)
    assert len(indices) <= 2
    # When filtering on error only (minimize), row 0 (error=1) is best
    assert 0 in indices


def test_top_k_pareto_filter_k_larger_than_n() -> None:
    observables = [Observable("m", Direction.MINIMIZE)]
    rt = ResultsTable(
        configs=[{"a": 1}, {"a": 2}],
        scores=np.array([[1.0], [2.0]]),
        observable_names=["m"],
    )
    fn = top_k_pareto_filter(k=10)
    indices = fn(rt, observables)
    assert len(indices) == 2


# ---------------------------------------------------------------------------
# Phase chaining with filter_fn (#21)
# ---------------------------------------------------------------------------


def test_phase_chaining_two_phases(
    world: _ToySimulator,
    scorer: _ToyScorer,
    grid: list[dict[str, Any]],
    observables: list[Observable],
) -> None:
    study = Study(
        world=world,
        scorer=scorer,
        observables=observables,
        phases=[
            Phase(
                name="discovery",
                grid=grid,
                filter_fn=top_k_pareto_filter(k=3),
            ),
            Phase(name="refinement", grid="previous"),
        ],
    )
    study.run()
    disc = study.results("discovery")
    ref = study.results("refinement")
    assert len(disc.configs) == 5
    assert len(ref.configs) <= 3


def test_phase_chaining_carry_grid(
    world: _ToySimulator,
    scorer: _ToyScorer,
    observables: list[Observable],
) -> None:
    """Second phase with non-list grid uses filtered configs from first."""
    grid = [{"alpha": v} for v in [0.0, 0.25, 0.5, 0.75, 1.0]]
    study = Study(
        world=world,
        scorer=scorer,
        observables=observables,
        phases=[
            Phase(
                name="phase1",
                grid=grid,
                filter_fn=top_k_pareto_filter(k=2),
            ),
            Phase(name="phase2", grid="previous"),
        ],
    )
    study.run()
    p2 = study.results("phase2")
    # Phase2 re-evaluates the filtered configs
    assert len(p2.configs) == 2


def test_phase_chaining_terminal_phase(
    world: _ToySimulator,
    scorer: _ToyScorer,
    grid: list[dict[str, Any]],
    observables: list[Observable],
) -> None:
    """Terminal phase (filter_fn=None) does not carry configs."""
    study = Study(
        world=world,
        scorer=scorer,
        observables=observables,
        phases=[Phase(name="only", grid=grid)],
    )
    study.run()
    r = study.results("only")
    assert len(r.configs) == 5


def test_phase_chaining_with_annotations(
    world: _ToySimulator,
    scorer: _ToyScorer,
    observables: list[Observable],
) -> None:
    grid = [
        {"alpha": 0.1, "method": "a"},
        {"alpha": 0.5, "method": "b"},
        {"alpha": 0.9, "method": "a"},
    ]
    annotations = [
        Annotation(name="method_cost", lookup={"a": 10.0, "b": 20.0}, key="method"),
    ]
    study = Study(
        world=world,
        scorer=scorer,
        observables=observables,
        phases=[Phase(name="p1", grid=grid)],
        annotations=annotations,
    )
    study.run()
    r = study.results("p1")
    assert r.annotations is not None
    assert r.annotations.shape == (3, 1)


def test_phase_chaining_scores_correct(
    world: _ToySimulator,
    scorer: _ToyScorer,
    observables: list[Observable],
) -> None:
    grid = [{"alpha": 0.5}]
    study = Study(
        world=world,
        scorer=scorer,
        observables=observables,
        phases=[Phase(name="exact", grid=grid)],
    )
    study.run()
    r = study.results("exact")
    assert r.scores[0, 0] == pytest.approx(0.0)  # error = |0.5 - 0.5|
    assert r.scores[0, 1] == pytest.approx(5.0)  # cost = 0.5 * 10


# ---------------------------------------------------------------------------
# Study.summary() (#23)
# ---------------------------------------------------------------------------


def test_summary_keys(
    world: _ToySimulator,
    scorer: _ToyScorer,
    grid: list[dict[str, Any]],
    observables: list[Observable],
) -> None:
    study = Study(
        world=world,
        scorer=scorer,
        observables=observables,
        phases=[Phase(name="p1", grid=grid)],
    )
    study.run()
    s = study.summary()
    assert "p1" in s
    assert set(s["p1"].keys()) == {"n_trials", "n_front", "observable_ranges"}


def test_summary_n_trials(
    world: _ToySimulator,
    scorer: _ToyScorer,
    grid: list[dict[str, Any]],
    observables: list[Observable],
) -> None:
    study = Study(
        world=world,
        scorer=scorer,
        observables=observables,
        phases=[Phase(name="p1", grid=grid)],
    )
    study.run()
    s = study.summary()
    assert s["p1"]["n_trials"] == 5


def test_summary_observable_ranges(
    world: _ToySimulator,
    scorer: _ToyScorer,
    grid: list[dict[str, Any]],
    observables: list[Observable],
) -> None:
    study = Study(
        world=world,
        scorer=scorer,
        observables=observables,
        phases=[Phase(name="p1", grid=grid)],
    )
    study.run()
    s = study.summary()
    ranges = s["p1"]["observable_ranges"]
    assert "error" in ranges
    assert "cost" in ranges
    assert ranges["error"]["min"] == pytest.approx(0.0)
    assert ranges["error"]["max"] == pytest.approx(0.5)
    assert ranges["cost"]["min"] == pytest.approx(0.0)
    assert ranges["cost"]["max"] == pytest.approx(10.0)


def test_summary_n_front(
    world: _ToySimulator,
    scorer: _ToyScorer,
    grid: list[dict[str, Any]],
    observables: list[Observable],
) -> None:
    study = Study(
        world=world,
        scorer=scorer,
        observables=observables,
        phases=[Phase(name="p1", grid=grid)],
    )
    study.run()
    s = study.summary()
    assert s["p1"]["n_front"] >= 1


def test_summary_multi_phase(
    world: _ToySimulator,
    scorer: _ToyScorer,
    grid: list[dict[str, Any]],
    observables: list[Observable],
) -> None:
    study = Study(
        world=world,
        scorer=scorer,
        observables=observables,
        phases=[
            Phase(
                name="disc",
                grid=grid,
                filter_fn=top_k_pareto_filter(k=3),
            ),
            Phase(name="refine", grid="previous"),
        ],
    )
    study.run()
    s = study.summary()
    assert "disc" in s
    assert "refine" in s
    assert s["refine"]["n_trials"] <= 3


# ---------------------------------------------------------------------------
# Study.front / Study.front_hypervolume / Study.stack
# ---------------------------------------------------------------------------


def test_front_returns_indices(
    world: _ToySimulator,
    scorer: _ToyScorer,
    grid: list[dict[str, Any]],
    observables: list[Observable],
) -> None:
    study = Study(
        world=world,
        scorer=scorer,
        observables=observables,
        phases=[Phase(name="p1", grid=grid)],
    )
    study.run()
    front_idx = study.front("p1")
    assert front_idx.dtype == np.intp
    assert len(front_idx) >= 1
    assert all(0 <= i < 5 for i in front_idx)


def test_front_hypervolume_positive(
    world: _ToySimulator,
    scorer: _ToyScorer,
    grid: list[dict[str, Any]],
    observables: list[Observable],
) -> None:
    study = Study(
        world=world,
        scorer=scorer,
        observables=observables,
        phases=[Phase(name="p1", grid=grid)],
    )
    study.run()
    ref = np.array([1.0, 20.0])
    hv = study.front_hypervolume("p1", ref)
    assert hv > 0.0


def test_stack_returns_weights(
    world: _ToySimulator,
    scorer: _ToyScorer,
    grid: list[dict[str, Any]],
    observables: list[Observable],
) -> None:
    study = Study(
        world=world,
        scorer=scorer,
        observables=observables,
        phases=[Phase(name="p1", grid=grid)],
    )
    study.run()
    weights = study.stack("p1")
    # stack_scores treats scores.T as (n_observables, n_configs)
    assert weights.shape == (2,)
    assert np.sum(weights) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Study.run() with adaptive phase (#25)
# ---------------------------------------------------------------------------


def test_adaptive_phase(
    world: _ToySimulator,
    scorer: _ToyScorer,
    observables: list[Observable],
) -> None:
    factors = [Factor("alpha", FactorType.CONTINUOUS, bounds=(0.0, 1.0))]
    study = Study(
        world=world,
        scorer=scorer,
        observables=observables,
        phases=[Phase(name="adaptive", grid="adaptive", n_trials=15)],
        factors=factors,
    )
    study.run()
    r = study.results("adaptive")
    assert len(r.configs) == 15
    assert r.scores.shape == (15, 2)


def test_adaptive_then_grid_phase(
    world: _ToySimulator,
    scorer: _ToyScorer,
    observables: list[Observable],
) -> None:
    factors = [Factor("alpha", FactorType.CONTINUOUS, bounds=(0.0, 1.0))]
    study = Study(
        world=world,
        scorer=scorer,
        observables=observables,
        phases=[
            Phase(
                name="explore",
                grid="adaptive",
                n_trials=10,
                filter_fn=top_k_pareto_filter(k=5),
            ),
            Phase(name="refine", grid="previous"),
        ],
        factors=factors,
    )
    study.run()
    explore_r = study.results("explore")
    refine_r = study.results("refine")
    assert len(explore_r.configs) == 10
    assert len(refine_r.configs) <= 5


def test_adaptive_summary(
    world: _ToySimulator,
    scorer: _ToyScorer,
    observables: list[Observable],
) -> None:
    factors = [Factor("alpha", FactorType.CONTINUOUS, bounds=(0.0, 1.0))]
    study = Study(
        world=world,
        scorer=scorer,
        observables=observables,
        phases=[Phase(name="a", grid="adaptive", n_trials=10)],
        factors=factors,
    )
    study.run()
    s = study.summary()
    assert s["a"]["n_trials"] == 10


# ---------------------------------------------------------------------------
# Callable grid mode (#94)
# ---------------------------------------------------------------------------


def _double_grid(
    results: ResultsTable,
    _observables: list[Observable],
) -> list[dict[str, Any]]:
    """Grid builder that creates a finer grid around carried configs.

    Returns:
        Doubled config list.
    """
    return [{"alpha": c["alpha"]} for c in results.configs] + [
        {"alpha": c["alpha"] + 0.01} for c in results.configs
    ]


def test_callable_grid_generates_new_configs(
    world: _ToySimulator,
    scorer: _ToyScorer,
    observables: list[Observable],
) -> None:
    grid = [{"alpha": v} for v in [0.0, 0.25, 0.5, 0.75, 1.0]]
    study = Study(
        world=world,
        scorer=scorer,
        observables=observables,
        phases=[
            Phase(
                name="discovery",
                grid=grid,
                filter_fn=top_k_pareto_filter(k=3),
            ),
            Phase(name="refinement", grid=_double_grid),
        ],
    )
    study.run()
    r2 = study.results("refinement")
    # _double_grid doubles the full Phase 1 results (5 → 10)
    assert len(r2.configs) == 10


def test_callable_grid_receives_filtered_results(
    world: _ToySimulator,
    scorer: _ToyScorer,
    observables: list[Observable],
) -> None:
    """Callable grid receives the full Phase 1 results (not filtered)."""
    received: list[ResultsTable] = []

    def capture_grid(
        results: ResultsTable,
        _obs: list[Observable],
    ) -> list[dict[str, Any]]:
        received.append(results)
        return [{"alpha": 0.5}]

    grid = [{"alpha": v} for v in [0.0, 0.25, 0.5, 0.75, 1.0]]
    study = Study(
        world=world,
        scorer=scorer,
        observables=observables,
        phases=[
            Phase(name="p1", grid=grid, filter_fn=top_k_pareto_filter(k=3)),
            Phase(name="p2", grid=capture_grid),
        ],
    )
    study.run()
    # The callable receives Phase 1's full results
    assert len(received) == 1
    assert len(received[0].configs) == 5


def test_callable_grid_on_first_phase_raises(
    world: _ToySimulator,
    scorer: _ToyScorer,
    observables: list[Observable],
) -> None:
    def dummy_grid(
        _results: ResultsTable,
        _obs: list[Observable],
    ) -> list[dict[str, Any]]:
        return []  # pragma: no cover

    study = Study(
        world=world,
        scorer=scorer,
        observables=observables,
        phases=[Phase(name="bad", grid=dummy_grid)],
    )
    with pytest.raises(ValueError, match="callable grid requires a previous phase"):
        study.run()


def test_callable_grid_with_three_phases(
    world: _ToySimulator,
    scorer: _ToyScorer,
    observables: list[Observable],
) -> None:
    """Callable grid works in a three-phase chain."""

    def narrow_grid(
        _results: ResultsTable,
        _obs: list[Observable],
    ) -> list[dict[str, Any]]:
        return [{"alpha": 0.4}, {"alpha": 0.5}, {"alpha": 0.6}]

    grid = [{"alpha": v} for v in [0.0, 0.25, 0.5, 0.75, 1.0]]
    study = Study(
        world=world,
        scorer=scorer,
        observables=observables,
        phases=[
            Phase(
                name="broad",
                grid=grid,
                filter_fn=top_k_pareto_filter(k=3),
            ),
            Phase(
                name="narrow",
                grid=narrow_grid,
                filter_fn=top_k_pareto_filter(k=2),
            ),
            Phase(name="final", grid="carry"),
        ],
    )
    study.run()
    assert len(study.results("broad").configs) == 5
    assert len(study.results("narrow").configs) == 3
    assert len(study.results("final").configs) <= 2


# ---------------------------------------------------------------------------
# weighted_sum_filter (#91)
# ---------------------------------------------------------------------------


def test_weighted_sum_filter_returns_callable() -> None:
    fn = weighted_sum_filter(weights={"error": 0.8, "cost": 0.2}, k=3)
    assert callable(fn)


def test_weighted_sum_filter_keeps_at_most_k(
    observables: list[Observable],
) -> None:
    rt = ResultsTable(
        configs=[{"alpha": v} for v in [0.0, 0.25, 0.5, 0.75, 1.0]],
        scores=np.array([
            [0.5, 0.0],
            [0.25, 2.5],
            [0.0, 5.0],
            [0.25, 7.5],
            [0.5, 10.0],
        ]),
        observable_names=["error", "cost"],
    )
    fn = weighted_sum_filter(weights={"error": 0.5, "cost": 0.5}, k=3)
    indices = fn(rt, observables)
    assert len(indices) <= 3


def test_weighted_sum_filter_respects_weights(
    observables: list[Observable],
) -> None:
    """Heavy weight on error should prefer configs with low error."""
    rt = ResultsTable(
        configs=[{"alpha": v} for v in [0.0, 0.25, 0.5, 0.75, 1.0]],
        scores=np.array([
            [0.5, 0.0],  # idx 0: high error, low cost
            [0.25, 2.5],  # idx 1
            [0.0, 5.0],  # idx 2: zero error, mid cost
            [0.25, 7.5],  # idx 3
            [0.5, 10.0],  # idx 4: high error, high cost
        ]),
        observable_names=["error", "cost"],
    )
    fn = weighted_sum_filter(weights={"error": 0.99, "cost": 0.01}, k=1)
    indices = fn(rt, observables)
    # Config with error=0.0 (idx 2) should be the best
    assert indices[0] == 2


def test_weighted_sum_filter_maximize_direction() -> None:
    """MAXIMIZE objectives are flipped so higher is better."""
    obs = [
        Observable("quality", Direction.MAXIMIZE),
        Observable("cost", Direction.MINIMIZE),
    ]
    rt = ResultsTable(
        configs=[{"a": 1}, {"a": 2}, {"a": 3}],
        scores=np.array([
            [10.0, 1.0],  # high quality, low cost → best
            [5.0, 5.0],  # mid
            [1.0, 10.0],  # low quality, high cost → worst
        ]),
        observable_names=["quality", "cost"],
    )
    fn = weighted_sum_filter(weights={"quality": 0.5, "cost": 0.5}, k=1)
    indices = fn(rt, obs)
    assert indices[0] == 0


def test_weighted_sum_filter_subset_objectives() -> None:
    """Only named objectives are used for ranking."""
    obs = [
        Observable("error", Direction.MINIMIZE),
        Observable("cost", Direction.MINIMIZE),
    ]
    rt = ResultsTable(
        configs=[{"a": 1}, {"a": 2}],
        scores=np.array([
            [1.0, 100.0],  # low error, very high cost
            [2.0, 1.0],  # higher error, low cost
        ]),
        observable_names=["error", "cost"],
    )
    # Only weight error → idx 0 (error=1) is best despite huge cost
    fn = weighted_sum_filter(weights={"error": 1.0}, k=1)
    indices = fn(rt, obs)
    assert indices[0] == 0


def test_weighted_sum_filter_constant_column() -> None:
    """Constant columns don't cause division by zero."""
    obs = [Observable("m", Direction.MINIMIZE)]
    rt = ResultsTable(
        configs=[{"a": 1}, {"a": 2}],
        scores=np.array([[5.0], [5.0]]),
        observable_names=["m"],
    )
    fn = weighted_sum_filter(weights={"m": 1.0}, k=2)
    indices = fn(rt, obs)
    assert len(indices) == 2


# ---------------------------------------------------------------------------
# Observable.weight propagation (#90)
# ---------------------------------------------------------------------------


def test_front_uses_observable_weights() -> None:
    """Study.front() propagates weights from Observable."""
    world = _ToySimulator()
    scorer = _ToyScorer()
    obs = [
        Observable("error", Direction.MINIMIZE, weight=2.0),
        Observable("cost", Direction.MINIMIZE, weight=1.0),
    ]
    grid = [{"alpha": v} for v in [0.0, 0.25, 0.5, 0.75, 1.0]]
    study = Study(
        world=world, scorer=scorer, observables=obs, phases=[Phase("p", grid)]
    )
    study.run()
    front_idx = study.front("p")
    assert front_idx.dtype == np.intp
    assert len(front_idx) >= 1


def test_front_hypervolume_uses_weights() -> None:
    """Study.front_hypervolume() propagates weights."""
    world = _ToySimulator()
    scorer = _ToyScorer()
    obs = [
        Observable("error", Direction.MINIMIZE, weight=2.0),
        Observable("cost", Direction.MINIMIZE, weight=1.0),
    ]
    grid = [{"alpha": v} for v in [0.0, 0.25, 0.5, 0.75, 1.0]]
    study = Study(
        world=world, scorer=scorer, observables=obs, phases=[Phase("p", grid)]
    )
    study.run()
    hv = study.front_hypervolume("p", np.array([2.0, 20.0]))
    assert hv > 0.0


def test_weighted_sum_filter_in_phase(
    world: _ToySimulator,
    scorer: _ToyScorer,
) -> None:
    """weighted_sum_filter works as Phase.filter_fn in a Study."""
    obs = [
        Observable("error", Direction.MINIMIZE),
        Observable("cost", Direction.MINIMIZE),
    ]
    grid = [{"alpha": v} for v in [0.0, 0.25, 0.5, 0.75, 1.0]]
    study = Study(
        world=world,
        scorer=scorer,
        observables=obs,
        phases=[
            Phase(
                name="disc",
                grid=grid,
                filter_fn=weighted_sum_filter(
                    weights={"error": 0.7, "cost": 0.3},
                    k=2,
                ),
            ),
            Phase(name="refine", grid="carry"),
        ],
    )
    study.run()
    assert len(study.results("refine").configs) == 2


# ---------------------------------------------------------------------------
# Study.run() progress callback (#77)
# ---------------------------------------------------------------------------


def test_study_run_callback(
    world: _ToySimulator,
    scorer: _ToyScorer,
    observables: list[Observable],
) -> None:
    """Study.run(callback=...) invokes callback for each trial."""
    grid = [{"alpha": v} for v in [0.0, 0.25, 0.5]]
    calls: list[tuple[int, int, TrialResult]] = []
    study = Study(
        world=world,
        scorer=scorer,
        observables=observables,
        phases=[Phase(name="p1", grid=grid)],
    )
    study.run(callback=lambda i, n, r: calls.append((i, n, r)))
    assert len(calls) == 3


def test_study_run_callback_multi_phase(
    world: _ToySimulator,
    scorer: _ToyScorer,
    observables: list[Observable],
) -> None:
    """Callback fires across multiple grid phases."""
    grid = [{"alpha": v} for v in [0.0, 0.25, 0.5, 0.75, 1.0]]
    calls: list[tuple[int, int, TrialResult]] = []
    study = Study(
        world=world,
        scorer=scorer,
        observables=observables,
        phases=[
            Phase(
                name="disc",
                grid=grid,
                filter_fn=top_k_pareto_filter(k=2),
            ),
            Phase(name="refine", grid="carry"),
        ],
    )
    study.run(callback=lambda i, n, r: calls.append((i, n, r)))
    # 5 trials in phase 1 + 2 trials in phase 2
    assert len(calls) == 7


# ---------------------------------------------------------------------------
# feasibility_filter (#74)
# ---------------------------------------------------------------------------


def test_feasibility_filter_returns_callable() -> None:
    fn = feasibility_filter(constraints=[])
    assert callable(fn)


def test_feasibility_filter_keeps_feasible(
    observables: list[Observable],
) -> None:
    rt = ResultsTable(
        configs=[{"alpha": v} for v in [0.0, 0.25, 0.5, 0.75, 1.0]],
        scores=np.array([
            [0.5, 0.0],
            [0.25, 2.5],
            [0.0, 5.0],
            [0.25, 7.5],
            [0.5, 10.0],
        ]),
        observable_names=["error", "cost"],
    )
    fn = feasibility_filter([
        Constraint("low_error", "error", "<=", 0.25),
    ])
    idx = fn(rt, observables)
    assert set(idx.tolist()) == {1, 2, 3}


def test_feasibility_filter_multiple_constraints(
    observables: list[Observable],
) -> None:
    rt = ResultsTable(
        configs=[{"alpha": v} for v in [0.0, 0.25, 0.5, 0.75, 1.0]],
        scores=np.array([
            [0.5, 0.0],
            [0.25, 2.5],
            [0.0, 5.0],
            [0.25, 7.5],
            [0.5, 10.0],
        ]),
        observable_names=["error", "cost"],
    )
    fn = feasibility_filter([
        Constraint("low_error", "error", "<=", 0.25),
        Constraint("low_cost", "cost", "<=", 5.0),
    ])
    idx = fn(rt, observables)
    assert set(idx.tolist()) == {1, 2}


def test_feasibility_filter_none_feasible(
    observables: list[Observable],
) -> None:
    rt = ResultsTable(
        configs=[{"alpha": 0.0}],
        scores=np.array([[0.5, 0.0]]),
        observable_names=["error", "cost"],
    )
    fn = feasibility_filter([
        Constraint("impossible", "error", "<", 0.0),
    ])
    idx = fn(rt, observables)
    assert len(idx) == 0


def test_feasibility_filter_in_study(
    world: _ToySimulator,
    scorer: _ToyScorer,
    observables: list[Observable],
) -> None:
    """feasibility_filter works as a Phase.filter_fn in a Study."""
    grid = [{"alpha": v} for v in [0.0, 0.25, 0.5, 0.75, 1.0]]
    study = Study(
        world=world,
        scorer=scorer,
        observables=observables,
        phases=[
            Phase(
                name="screen",
                grid=grid,
                filter_fn=feasibility_filter([
                    Constraint("low_cost", "cost", "<=", 5.0),
                ]),
            ),
            Phase(name="refine", grid="carry"),
        ],
    )
    study.run()
    final = study.results("refine")
    # alpha=0.0 (cost=0), alpha=0.25 (cost=2.5), alpha=0.5 (cost=5.0) satisfy cost <= 5
    assert final.scores.shape[0] == 3
