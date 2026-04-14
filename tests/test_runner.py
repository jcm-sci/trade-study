"""Tests for runner module (issues #18, #19, #20)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from trade_study.design import Factor, FactorType
from trade_study.protocols import Annotation, Direction, Observable
from trade_study.runner import run_adaptive, run_grid

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
