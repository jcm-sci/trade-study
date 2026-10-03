"""Evaluation-level persistence, failure recovery, and bounded retry tests."""

from __future__ import annotations

import sqlite3
import time
from contextlib import closing
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from joblib import parallel_backend

from trade_study import Annotation, Direction, Observable, Phase, Study, run_grid

if TYPE_CHECKING:
    from pathlib import Path

_GRID = [{"x": 0}, {"x": 1}, {"x": 2}]
_OBS = [Observable("loss", Direction.MINIMIZE)]


class _World:
    def __init__(self, fail: tuple[int, int] | None = None) -> None:
        self.fail = fail
        self.calls: list[tuple[int, int]] = []

    def generate(self, config: dict[str, Any], *, rep: int = 0) -> tuple[int, int]:
        task = config["x"], rep
        self.calls.append(task)
        if task == self.fail:
            msg = "simulator failed"
            raise RuntimeError(msg)
        return config["x"], rep


class _Scorer:
    def __init__(self, failures: int = 0) -> None:
        self.failures = failures

    def score(self, truth: int, observations: int, config: dict[str, Any]) -> dict:
        if self.failures:
            self.failures -= 1
            msg = "scorer failed"
            raise LookupError(msg)
        return {"loss": float(truth + observations)}


def test_resume_keeps_completed_replicates_and_original_order(tmp_path: Path) -> None:
    path = tmp_path / "grid.sqlite"
    world = _World(fail=(1, 1))
    with pytest.raises(RuntimeError, match="simulator failed"):
        run_grid(world, _Scorer(), _GRID, _OBS, n_reps=2, checkpoint_path=path)
    assert world.calls == [(0, 0), (0, 1), (1, 0), (1, 1)]

    restarted = _World()
    callbacks = []
    result = run_grid(
        restarted,
        _Scorer(),
        _GRID,
        _OBS,
        n_reps=2,
        checkpoint_path=path,
        callback=lambda i, total, trial: callbacks.append((i, total, trial.config)),
    )
    assert restarted.calls == [(1, 1), (2, 0), (2, 1)]
    assert result.configs == [cfg for cfg in _GRID for _ in range(2)]
    assert [(m["design_point"], m["rep"]) for m in result.metadata] == [
        (x, rep) for x in range(3) for rep in range(2)
    ]
    assert [m.get("recovered", False) for m in result.metadata] == [True] * 3 + [
        False
    ] * 3
    assert result.metadata[3]["attempts"] == 2
    assert [i for i, _, _ in callbacks] == list(range(6))
    np.testing.assert_array_equal(result.scores[:, 0], [0, 1, 1, 2, 2, 3])


def test_callback_interruption_keeps_completed_evaluation(tmp_path: Path) -> None:
    path = tmp_path / "callback.sqlite"

    def interrupt(index: int, total: int, trial: object) -> None:
        del index, total, trial
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_grid(
            _World(), _Scorer(), _GRID, _OBS, checkpoint_path=path, callback=interrupt
        )
    reopened = _World()
    run_grid(reopened, _Scorer(), _GRID, _OBS, checkpoint_path=path)
    assert reopened.calls == [(1, 0), (2, 0)]


@pytest.mark.parametrize("persistent", [False, True])
def test_scorer_retry_is_opt_in_and_retains_replicate_identity(
    tmp_path: Path, *, persistent: bool
) -> None:
    path = tmp_path / "retry.sqlite" if persistent else None
    world = _World()
    result = run_grid(
        world,
        _Scorer(failures=1),
        _GRID[:1],
        _OBS,
        max_retries=1,
        checkpoint_path=path,
    )
    assert world.calls == [(0, 0), (0, 0)]
    assert result.metadata[0]["attempts"] == 2
    with pytest.raises(LookupError, match="scorer failed"):
        run_grid(_World(), _Scorer(failures=1), _GRID[:1], _OBS)
    limited = _World()
    with pytest.raises(LookupError, match="scorer failed"):
        run_grid(limited, _Scorer(failures=3), _GRID[:1], _OBS, max_retries=1)
    assert len(limited.calls) == 2
    with pytest.raises(ValueError, match="max_retries"):
        run_grid(_World(), _Scorer(), _GRID, _OBS, max_retries=-1)


@pytest.mark.parametrize("change", ["grid", "reps", "schema", "revision", "annotation"])
def test_incompatible_grid_checkpoint_is_rejected(tmp_path: Path, change: str) -> None:
    path = tmp_path / "schema.sqlite"
    run_grid(
        _World(), _Scorer(), _GRID, _OBS, checkpoint_path=path, checkpoint_key="v1"
    )
    grid = _GRID[::-1] if change == "grid" else _GRID
    reps = 2 if change == "reps" else 1
    obs = [Observable("loss", Direction.MAXIMIZE)] if change == "schema" else _OBS
    key = "v2" if change == "revision" else "v1"
    annotations = [Annotation("cost", float, "x")] if change == "annotation" else None
    with pytest.raises(ValueError, match="Incompatible grid checkpoint"):
        run_grid(
            _World(),
            _Scorer(),
            grid,
            obs,
            n_reps=reps,
            annotations=annotations,
            checkpoint_path=path,
            checkpoint_key=key,
        )


class _ParallelWorld(_World):
    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = path
        self.fail_worker = True

    def generate(self, config: dict[str, Any], *, rep: int = 0) -> tuple[int, int]:
        if config["x"] == 1 and self.fail_worker:
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                with closing(sqlite3.connect(self.path)) as connection:
                    completed = connection.execute(
                        "SELECT COUNT(*) FROM trials WHERE payload IS NOT NULL"
                    ).fetchone()[0]
                if completed:
                    msg = "parallel worker failed"
                    raise RuntimeError(msg)
                time.sleep(0.01)
            msg = "completed worker did not persist its result"
            raise TimeoutError(msg)
        return super().generate(config, rep=rep)


def test_parallel_worker_saves_before_another_worker_fails(tmp_path: Path) -> None:
    path = tmp_path / "parallel.sqlite"
    world = _ParallelWorld(path)
    with parallel_backend("threading"), pytest.raises(RuntimeError, match="parallel"):
        run_grid(world, _Scorer(), _GRID[:2], _OBS, n_jobs=2, checkpoint_path=path)
    # Keep the same simulator class and revision; stop its injected failure.
    grid = _GRID[:2]
    with closing(sqlite3.connect(path)) as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM trials WHERE payload IS NOT NULL"
            ).fetchone()[0]
            == 1
        )
    # Resume through the public runner without re-evaluating the completed task.
    world.fail_worker = False
    world.calls.clear()
    with parallel_backend("threading"):
        result = run_grid(world, _Scorer(), grid, _OBS, n_jobs=2, checkpoint_path=path)
    assert world.calls == [(1, 0)]
    np.testing.assert_array_equal(result.scores[:, 0], [0, 1])


def test_study_recovers_within_an_incomplete_phase(tmp_path: Path) -> None:
    first = Study(
        _World(fail=(1, 1)), _Scorer(), _OBS, phases=[Phase("grid", _GRID, n_reps=2)]
    )
    with pytest.raises(RuntimeError, match="simulator failed"):
        first.run(checkpoint_dir=tmp_path)
    restarted = _World()
    second = Study(restarted, _Scorer(), _OBS, phases=[Phase("grid", _GRID, n_reps=2)])
    second.run(checkpoint_dir=tmp_path)
    assert restarted.calls == [(1, 1), (2, 0), (2, 1)]
    assert second.results("grid").scores.shape == (6, 1)


def test_empty_grid_keeps_matrix_shapes(tmp_path: Path) -> None:
    result = run_grid(
        _World(),
        _Scorer(),
        [],
        _OBS,
        annotations=[Annotation("cost", {}, "x")],
        checkpoint_path=tmp_path / "empty.sqlite",
    )
    assert result.scores.shape == (0, 1)
    assert result.annotations is not None
    assert result.annotations.shape == (0, 1)


def test_process_workers_can_share_the_ledger(tmp_path: Path) -> None:
    path = tmp_path / "processes.sqlite"
    first = run_grid(_World(), _Scorer(), _GRID, _OBS, n_jobs=2, checkpoint_path=path)
    second = run_grid(_World(), _Scorer(), _GRID, _OBS, n_jobs=2, checkpoint_path=path)
    np.testing.assert_array_equal(first.scores, second.scores)
    assert all(m["recovered"] for m in second.metadata)
