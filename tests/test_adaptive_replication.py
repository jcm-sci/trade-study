"""Integration regressions for adaptive phase replication (#142)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from trade_study import (
    Constraint,
    Direction,
    Factor,
    FactorType,
    Observable,
    Phase,
    Study,
    run_adaptive,
    top_k_pareto_filter,
)

if TYPE_CHECKING:
    from pathlib import Path

FACTORS = [Factor("x", FactorType.CONTINUOUS, bounds=(0.0, 1.0))]
OBSERVABLES = [Observable("loss", Direction.MINIMIZE, weight=2.0)]


class _World:
    def __init__(self) -> None:
        self.reps: list[int] = []

    def generate(self, config: dict[str, Any], *, rep: int = 0) -> tuple[float, None]:
        self.reps.append(rep)
        return float(config["x"]) + rep, None


class _Scorer:
    def score(
        self, truth: float, observations: None, config: dict[str, Any]
    ) -> dict[str, float]:
        return {"loss": truth}


def _study(world: _World) -> Study:
    return Study(
        world=world,
        scorer=_Scorer(),
        factors=FACTORS,
        observables=OBSERVABLES,
        phases=[
            Phase(
                "search",
                grid="adaptive",
                n_trials=4,
                n_reps=3,
                filter_fn=top_k_pareto_filter(2),
            ),
            Phase("verify", grid="carry", n_reps=2),
        ],
    )


def test_runner_keeps_raw_mean_error_and_count() -> None:
    world = _World()
    table = run_adaptive(world, _Scorer(), FACTORS, OBSERVABLES, n_trials=2, n_reps=3)
    assert world.reps == [0, 1, 2] * 2
    for config, score, meta in zip(
        table.configs, table.scores, table.metadata, strict=True
    ):
        assert meta["scores"]["loss"] == pytest.approx(config["x"] + 1)
        assert score[0] == pytest.approx(2 * meta["scores"]["loss"])
        assert meta["standard_error"]["loss"] == pytest.approx(1 / np.sqrt(3))
        assert meta["n_reps"]["loss"] == 3


def test_adaptive_filter_comparison_and_checkpoint_round_trip(tmp_path: Path) -> None:
    world = _World()
    study = _study(world)
    study.run(checkpoint_dir=tmp_path)
    assert world.reps == [0, 1, 2] * 4 + [0, 1] * 2
    assert len(study.results("search").configs) == 4
    assert len(study.results("verify").configs) == 4
    assert [row["n_trials"] for row in study.compare_phases()] == [4, 2]
    kept = top_k_pareto_filter(2)(study.results("search"), OBSERVABLES)
    assert study.results("verify").configs[::2] == [
        study.results("search").configs[i] for i in kept
    ]
    restored_world = _World()
    restored = _study(restored_world)
    restored.run(checkpoint_dir=tmp_path)
    assert restored_world.reps == []
    assert restored.results("search").metadata == study.results("search").metadata
    assert [row["n_trials"] for row in restored.compare_phases()] == [4, 2]


def test_weighted_results_constraints_use_raw_observable_units() -> None:
    table = run_adaptive(
        _World(), _Scorer(), FACTORS, OBSERVABLES, n_trials=2, n_reps=3
    )
    raw = np.array([meta["scores"]["loss"] for meta in table.metadata])
    threshold = float(raw[0] + 0.01)
    np.testing.assert_array_equal(
        table.feasible([Constraint("cap", "loss", "<=", threshold)]),
        raw <= threshold,
    )
    confident = Constraint("cap", "loss", "<=", threshold, confidence=0.95)
    expected = np.array([
        confident.check(
            confident.bound(meta["scores"]["loss"], meta["standard_error"]["loss"])
        )
        for meta in table.metadata
    ])
    np.testing.assert_array_equal(table.feasible([confident]), expected)
