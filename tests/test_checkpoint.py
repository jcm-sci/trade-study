"""Tests for study checkpointing and resumption (#75)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from trade_study.io import save_results
from trade_study.protocols import Direction, Observable, ResultsTable
from trade_study.study import Phase, Study, top_k_pareto_filter

if TYPE_CHECKING:
    from pathlib import Path

OBSERVABLES = [Observable("error", Direction.MINIMIZE)]
GRID = [{"alpha": a} for a in (0.1, 0.3, 0.5, 0.7, 0.9)]


class _World:
    def __init__(self, fail_on: float | None = None) -> None:
        self.calls = 0
        self.fail_on = fail_on

    def generate(self, config: dict[str, Any]) -> tuple[Any, Any]:
        self.calls += 1
        if self.fail_on is not None and config["alpha"] == self.fail_on:
            msg = "simulated crash"
            raise RuntimeError(msg)
        return config, config


class _Scorer:
    @staticmethod
    def score(truth: Any, observations: Any, config: dict[str, Any]) -> dict:
        del truth, observations
        return {"error": abs(config["alpha"] - 0.5)}


def _study(world: _World) -> Study:
    return Study(
        world=world,
        scorer=_Scorer(),
        observables=OBSERVABLES,
        phases=[
            Phase("screen", grid=GRID, filter_fn=top_k_pareto_filter(2)),
            Phase("refine", grid="carry"),
        ],
    )


def test_checkpoint_saves_every_phase(tmp_path: Path) -> None:
    study = _study(_World())

    study.run(checkpoint_dir=tmp_path)

    assert (tmp_path / "study.json").exists()
    assert (tmp_path / "00_screen" / "meta.json").exists()
    assert (tmp_path / "01_refine" / "meta.json").exists()


def test_rerun_loads_completed_phases_without_simulating(tmp_path: Path) -> None:
    first = _study(_World())
    first.run(checkpoint_dir=tmp_path)

    world = _World(fail_on=0.5)
    second = _study(world)
    second.run(checkpoint_dir=tmp_path)

    assert world.calls == 0
    for phase in ("screen", "refine"):
        np.testing.assert_array_equal(
            second.results(phase).scores, first.results(phase).scores
        )


def test_interrupted_study_resumes_from_the_last_completed_phase(
    tmp_path: Path,
) -> None:
    crashing = Study(
        world=_World(),
        scorer=_Scorer(),
        observables=OBSERVABLES,
        phases=[
            Phase("screen", grid=GRID, filter_fn=top_k_pareto_filter(2)),
            Phase("refine", grid="carry", world=_World(fail_on=0.5)),
        ],
    )
    with pytest.raises(RuntimeError, match="simulated crash"):
        crashing.run(checkpoint_dir=tmp_path)

    world = _World()
    resumed = _study(world)
    resumed.run(checkpoint_dir=tmp_path)

    assert world.calls == 2
    assert sorted(c["alpha"] for c in resumed.results("refine").configs) == [0.5, 0.7]


def test_externally_produced_phase_results_are_used(tmp_path: Path) -> None:
    external = ResultsTable(
        configs=[{"alpha": 0.3}, {"alpha": 0.6}],
        scores=np.array([[0.2], [0.1]]),
        observable_names=["error"],
    )
    study = _study(_World())
    study.save(tmp_path)
    save_results(external, tmp_path / "00_screen")

    world = _World()
    study = _study(world)
    study.run(checkpoint_dir=tmp_path)

    assert world.calls == 2
    assert [c["alpha"] for c in study.results("refine").configs] == [0.6, 0.3]


def test_checkpoint_of_another_study_is_refused(tmp_path: Path) -> None:
    _study(_World()).run(checkpoint_dir=tmp_path)
    other = Study(
        world=_World(),
        scorer=_Scorer(),
        observables=OBSERVABLES,
        phases=[Phase("different", grid=GRID)],
    )

    with pytest.raises(ValueError, match="written for phases"):
        other.run(checkpoint_dir=tmp_path)


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    study = _study(_World())
    study.run()
    study.save(tmp_path)

    restored = _study(_World())

    assert restored.load(tmp_path) == ["screen", "refine"]
    np.testing.assert_array_equal(
        restored.results("refine").scores, study.results("refine").scores
    )
    with pytest.raises(FileNotFoundError):
        restored.load(tmp_path / "missing")
