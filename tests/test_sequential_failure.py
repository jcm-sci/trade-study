"""Aborted-run diagnostics preserve failed identities and declared denominators."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from trade_study import (
    Direction,
    Observable,
    ReplicationPolicy,
    SequentialEvaluationError,
    load_result_bundle,
    run_sequential,
    save_result_bundle,
)

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Literal


class _World:
    def __init__(self, mode: str, fail_at: tuple[int, int] = (1, 1)) -> None:
        self.mode = mode
        self.fail_at = fail_at
        self.calls: list[tuple[int, int]] = []

    def generate(
        self, config: dict[str, Any], *, rep: int = 0
    ) -> tuple[int, tuple[int, int]]:
        key = (config["design"], rep)
        self.calls.append(key)
        if self.mode == "generate" and key == self.fail_at:
            msg = "generator failed"
            raise RuntimeError(msg)
        return config["design"], key


class _Scorer:
    def __init__(self, mode: str, fail_at: tuple[int, int] = (1, 1)) -> None:
        self.mode = mode
        self.fail_at = fail_at

    def score(
        self, truth: int, observations: tuple[int, int], config: dict[str, Any]
    ) -> dict[str, float]:
        del truth, config
        if observations != self.fail_at:
            return {"loss": 0.5}
        if self.mode == "score":
            msg = "scorer failed"
            raise OSError(msg)
        if self.mode == "missing":
            return {}
        return {
            "loss": {"nan": np.nan, "infinite": np.inf, "outside": 2.0}.get(
                self.mode, 0.5
            )
        }


@pytest.mark.parametrize("allocation", ["fixed", "adaptive"])
@pytest.mark.parametrize(
    "mode", ["generate", "score", "missing", "nan", "infinite", "outside"]
)
def test_abort_keeps_attempt_counts_and_raw_order(
    mode: str, allocation: Literal["fixed", "adaptive"], tmp_path: Path
) -> None:
    """A failed fourth call is retained separately from three valid scores."""
    world = _World(mode)
    grid = [{"design": 0, "nested": [1]}, {"design": 1, "nested": [2]}]
    observables = [Observable("loss", Direction.MINIMIZE)]
    policy = ReplicationPolicy({"loss": (0, 1)}, max_reps=4, allocation=allocation)
    with pytest.raises(SequentialEvaluationError) as caught:
        run_sequential(
            world, _Scorer(mode), grid, observables, policy=policy, capture_failure=True
        )
    failure = caught.value.failure
    expected_cause = (
        RuntimeError
        if mode == "generate"
        else OSError
        if mode == "score"
        else ValueError
    )
    assert isinstance(caught.value.__cause__, expected_cause)
    assert failure.stage == (
        "evaluation" if mode in {"generate", "score"} else "score_validation"
    )
    assert failure.completed_reps == (2, 1)
    assert (failure.failed_design, failure.failed_rep) == (1, 1)
    assert world.calls == [(0, 0), (1, 0), (0, 1), (1, 1)]
    assert failure.results.scores.shape == (3, 1)
    assert [row["evaluation_index"] for row in failure.results.metadata] == [0, 2, 1]
    assert all(
        row["run_status"] == "aborted" and row["decision_status"] == "unavailable"
        for row in failure.results.metadata
    )
    assert not any(
        hasattr(failure, name) for name in ("summary", "intervals", "feasibility")
    )
    metadata = failure.to_metadata()
    assert metadata["completed_reps"] == [2, 1]
    assert metadata["failed_reps"] == [0, 1]
    assert metadata["attempted_reps"] == [2, 2]
    assert metadata["actual_evaluations"] == 4
    assert metadata["completed_evaluations"] == 3
    assert metadata["unattempted_capacity"] == [2, 2]
    assert metadata["planned_reps"] == (4 if allocation == "fixed" else None)
    assert metadata["inference"] == "not_supplied"
    grid[0]["nested"].append(99)
    policy.score_bounds["loss"] = (-100, 100)
    assert failure.grid[0]["nested"] == [1]
    assert failure.results.configs[0]["nested"] == [1]
    assert failure.policy.score_bounds["loss"] == (0, 1)
    metadata["designs"][0]["nested"].append(20)
    assert failure.grid[0]["nested"] == [1]
    save_result_bundle(
        failure.results,
        tmp_path / "abort",
        observables=failure.observables,
        provenance={
            "sequential_failure": failure.to_metadata(),
            "random_stream": "design/replicate",
        },
    )
    loaded = load_result_bundle(tmp_path / "abort")
    assert loaded.provenance["caller"]["sequential_failure"] == failure.to_metadata()
    np.testing.assert_array_equal(loaded.results.scores, failure.results.scores)


def test_first_failure_round_trips_an_empty_diagnostic_table(tmp_path: Path) -> None:
    """No successful row is invented when the first evaluation fails."""
    with pytest.raises(SequentialEvaluationError) as caught:
        run_sequential(
            _World("generate", (0, 0)),
            _Scorer("generate"),
            [{"design": 0}, {"design": 1}],
            [Observable("loss", Direction.MINIMIZE)],
            policy=ReplicationPolicy({"loss": (0, 1)}),
            capture_failure=True,
        )
    failure = caught.value.failure
    assert failure.results.scores.shape == (0, 1)
    assert failure.completed_reps == (0, 0)
    assert failure.to_metadata()["attempted_reps"] == [1, 0]
    save_result_bundle(
        failure.results,
        tmp_path / "empty",
        observables=failure.observables,
        provenance={"sequential_failure": failure.to_metadata()},
    )
    loaded = load_result_bundle(tmp_path / "empty")
    assert loaded.results.scores.shape == (0, 1)
    assert loaded.provenance["caller"]["sequential_failure"]["failed_reps"] == [1, 0]


def test_capture_preserves_completed_run_scores_and_counts() -> None:
    """Successful capture mode changes no allocation or scores."""
    args = ([{"design": 0}, {"design": 1}], [Observable("loss", Direction.MINIMIZE)])
    policy = ReplicationPolicy({"loss": (0, 1)}, max_reps=2, allocation="fixed")
    ordinary = run_sequential(_World("okay"), _Scorer("okay"), *args, policy=policy)
    captured = run_sequential(
        _World("okay"), _Scorer("okay"), *args, policy=policy, capture_failure=True
    )
    np.testing.assert_array_equal(ordinary.results.scores, captured.results.scores)
    assert ordinary.allocations == captured.allocations == [2, 2]
    assert ordinary.stopping_reasons == captured.stopping_reasons


def test_default_does_not_wrap_evaluator_errors() -> None:
    """Existing callers retain their original error type and message."""
    with pytest.raises(RuntimeError, match="generator failed") as caught:
        run_sequential(
            _World("generate", (0, 0)),
            _Scorer("generate"),
            [{"design": 0}],
            [Observable("loss", Direction.MINIMIZE)],
            policy=ReplicationPolicy({"loss": (0, 1)}),
        )
    assert type(caught.value) is RuntimeError


@pytest.mark.parametrize("flag", [None, 1, "yes"])
def test_capture_flag_fails_before_any_evaluation(flag: Any) -> None:
    """Invalid capture flags cannot silently turn on diagnostic snapshots."""
    world = _World("okay")
    with pytest.raises(TypeError, match="capture_failure must be boolean"):
        run_sequential(
            world,
            _Scorer("okay"),
            [{"design": 0}],
            [Observable("loss", Direction.MINIMIZE)],
            policy=ReplicationPolicy({"loss": (0, 1)}),
            capture_failure=flag,
        )
    assert world.calls == []


def test_adaptive_failure_after_initial_batch_retains_all_prior_attempts() -> None:
    """A later ambiguous allocation cannot erase its completed initial batch."""
    world = _World("generate", (0, 2))
    with pytest.raises(SequentialEvaluationError) as caught:
        run_sequential(
            world,
            _Scorer("okay"),
            [{"design": 0}, {"design": 1}],
            [Observable("loss", Direction.MINIMIZE)],
            policy=ReplicationPolicy({"loss": (0, 1)}, max_reps=4),
            capture_failure=True,
        )
    failure = caught.value.failure
    assert failure.completed_reps == (2, 2)
    assert (failure.failed_design, failure.failed_rep) == (0, 2)
    assert failure.to_metadata()["actual_evaluations"] == 5
    assert failure.to_metadata()["attempted_reps"] == [3, 2]
    assert world.calls == [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2)]
