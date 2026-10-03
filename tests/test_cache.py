"""Evaluation reuse identities, invalidation, bypass, and independent replicates."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from trade_study import Annotation, Direction, EvaluationCache, Observable, run_grid

if TYPE_CHECKING:
    from pathlib import Path

_OBS = [Observable("loss", Direction.MINIMIZE)]
_GRID = [{"x": 0.5, "mode": "a"}, {"x": 1.0, "mode": "b"}]


class _World:
    def __init__(self, offset: float = 0) -> None:
        self.offset = offset
        self.calls: list[tuple[float, int]] = []

    def generate(self, config: dict[str, Any], *, rep: int = 0) -> tuple[float, float]:
        self.calls.append((config["x"], rep))
        return config["x"], config["x"] + rep + self.offset


class _Scorer:
    @staticmethod
    def score(truth: float, observations: float, config: dict[str, Any]) -> dict:
        del truth, config
        return {"loss": observations}


def _cache(path: Path, **changes: str) -> EvaluationCache:
    options = {
        "revision": "model-v1-scorer-v1",
        "replicate_namespace": "seed-17",
        "fidelity": "high",
    }
    options.update(changes)
    return EvaluationCache(path, **options)


def test_reordered_grid_and_increased_replication_reuse_exact_identities(
    tmp_path: Path,
) -> None:
    path = tmp_path / "cache.sqlite"
    world = _World()
    first = run_grid(world, _Scorer(), _GRID, _OBS, n_reps=2, cache=_cache(path))
    assert len(world.calls) == 4
    assert all(not m["cache_hit"] for m in first.metadata)
    world.calls.clear()
    reversed_grid = [dict(reversed(list(cfg.items()))) for cfg in _GRID[::-1]]
    second = run_grid(
        world, _Scorer(), reversed_grid, _OBS, n_reps=3, cache=_cache(path)
    )
    assert world.calls == [(1.0, 2), (0.5, 2)]
    assert [m["cache_hit"] for m in second.metadata] == [True, True, False] * 2
    assert [m["design_point"] for m in second.metadata] == [0, 0, 0, 1, 1, 1]
    assert second.metadata[0]["cache_context"]["replicate_namespace"] == "seed-17"
    np.testing.assert_array_equal(second.scores[:, 0], [1, 2, 3, 0.5, 1.5, 2.5])


@pytest.mark.parametrize("change", ["revision", "replicate_namespace", "fidelity"])
def test_behavior_or_independent_replication_changes_never_hit_old_entries(
    tmp_path: Path, change: str
) -> None:
    path = tmp_path / "context.sqlite"
    world = _World()
    run_grid(world, _Scorer(), _GRID, _OBS, cache=_cache(path))
    world.calls.clear()
    result = run_grid(
        world, _Scorer(), _GRID, _OBS, cache=_cache(path, **{change: "different"})
    )
    assert len(world.calls) == 2
    assert all(not m["cache_hit"] for m in result.metadata)


def test_objective_and_annotation_semantics_are_part_of_the_key(tmp_path: Path) -> None:
    cache = _cache(tmp_path / "schema.sqlite")
    world = _World()
    run_grid(world, _Scorer(), _GRID, _OBS, cache=cache)
    altered_obs = [Observable("loss", Direction.MAXIMIZE, weight=2)]
    result = run_grid(world, _Scorer(), _GRID, altered_obs, cache=cache)
    assert all(not m["cache_hit"] for m in result.metadata)
    costs = [Annotation("cost", {"a": 1, "b": 2}, "mode")]
    first = run_grid(world, _Scorer(), _GRID, _OBS, cache=cache, annotations=costs)
    costs[0].lookup["a"] = 3
    second = run_grid(world, _Scorer(), _GRID, _OBS, cache=cache, annotations=costs)
    assert first.metadata[0]["cache_key"] != second.metadata[0]["cache_key"]
    assert all(not m["cache_hit"] for m in second.metadata)
    assert second.annotations is not None
    assert second.annotations[0, 0] == 3


def test_bypass_does_not_replace_evidence_and_clear_invalidates(tmp_path: Path) -> None:
    cache = _cache(tmp_path / "bypass.sqlite")
    world = _World()
    first = run_grid(world, _Scorer(), _GRID, _OBS, cache=cache)
    world.offset = 10
    fresh = run_grid(world, _Scorer(), _GRID, _OBS, cache=cache, cache_bypass=True)
    np.testing.assert_array_equal(fresh.scores, first.scores + 10)
    assert all("cache_hit" not in m for m in fresh.metadata)
    cached = run_grid(world, _Scorer(), _GRID, _OBS, cache=cache)
    np.testing.assert_array_equal(cached.scores, first.scores)
    cache.clear()
    new = run_grid(world, _Scorer(), _GRID, _OBS, cache=cache)
    np.testing.assert_array_equal(new.scores, fresh.scores)


def test_completed_checkpoint_can_populate_cache_and_retains_identity(
    tmp_path: Path,
) -> None:
    path = tmp_path / "run.sqlite"
    cache = _cache(tmp_path / "evidence.sqlite")
    world = _World()
    run_grid(world, _Scorer(), _GRID, _OBS, checkpoint_path=path)
    world.calls.clear()
    recovered = run_grid(
        world, _Scorer(), _GRID, _OBS, checkpoint_path=path, cache=cache
    )
    assert world.calls == []
    assert all(m["recovered"] for m in recovered.metadata)
    result = run_grid(world, _Scorer(), _GRID[::-1], _OBS, cache=cache)
    assert world.calls == []
    assert all(m["cache_hit"] for m in result.metadata)


def test_parallel_workers_reuse_persistent_entries(tmp_path: Path) -> None:
    cache = _cache(tmp_path / "parallel.sqlite")
    first = run_grid(_World(), _Scorer(), _GRID, _OBS, cache=cache, n_jobs=2)
    second = run_grid(_World(), _Scorer(), _GRID, _OBS, cache=cache, n_jobs=2)
    np.testing.assert_array_equal(first.scores, second.scores)
    assert all(m["cache_hit"] for m in second.metadata)


class _Mode(Enum):
    A = "a"


def test_typed_canonicalization_does_not_collapse_container_or_key_types(
    tmp_path: Path,
) -> None:
    cache = _cache(tmp_path / "types.sqlite")
    configs = [
        {"x": 0.5, "extra": {1: "value"}},
        {"x": 0.5, "extra": {"1": "value"}},
        {"x": 0.5, "extra": [1]},
        {"x": 0.5, "extra": (1,)},
        {"x": 0.5, "extra": np.int64(1)},
        {"x": 0.5, "extra": _Mode.A},
    ]
    world = _World()
    first = run_grid(world, _Scorer(), configs, _OBS, cache=cache)
    assert len({m["cache_key"] for m in first.metadata}) == len(configs)
    world.calls.clear()
    run_grid(world, _Scorer(), configs, _OBS, cache=cache)
    assert world.calls == []
    with pytest.raises(ValueError, match="supported JSON/scalar"):
        run_grid(world, _Scorer(), [{"x": 0.5, "opaque": object()}], _OBS, cache=cache)


class _MutatingWorld(_World):
    def generate(self, config: dict[str, Any], *, rep: int = 0) -> tuple[float, float]:
        config["x"] += 1
        return super().generate(config, rep=rep)


def test_mutating_evaluator_cannot_cache_under_the_wrong_configuration(
    tmp_path: Path,
) -> None:
    grid = [{"x": 0.5}]
    with pytest.raises(ValueError, match="must not mutate"):
        run_grid(
            _MutatingWorld(),
            _Scorer(),
            grid,
            _OBS,
            cache=_cache(tmp_path / "mutation.sqlite"),
        )
    assert grid == [{"x": 0.5}]


def test_invalid_identities_and_cache_format_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "invalid.sqlite"
    with pytest.raises(ValueError, match="nonempty"):
        _cache(path, revision=" ")
    _cache(path)
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute("UPDATE cache_format SET version = 999")
    with pytest.raises(ValueError, match="Incompatible evaluation cache format"):
        _cache(path)


def test_conflicting_checkpoint_evidence_never_overwrites_cache(tmp_path: Path) -> None:
    checkpoint = tmp_path / "completed.sqlite"
    cache = _cache(tmp_path / "conflict.sqlite")
    run_grid(_World(), _Scorer(), _GRID, _OBS, checkpoint_path=checkpoint)
    # Intentionally lie about the revision to exercise conflict detection.
    newer = run_grid(_World(offset=10), _Scorer(), _GRID, _OBS, cache=cache)
    with pytest.raises(ValueError, match="Conflicting scores"):
        run_grid(
            _World(), _Scorer(), _GRID, _OBS, checkpoint_path=checkpoint, cache=cache
        )
    retained = run_grid(_World(), _Scorer(), _GRID, _OBS, cache=cache)
    np.testing.assert_array_equal(retained.scores, newer.scores)
