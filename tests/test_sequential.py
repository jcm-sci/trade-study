"""Budgeted replication under near ties, feasibility boundaries, and clear gaps."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from trade_study import (
    Constraint,
    Direction,
    Observable,
    ReplicationPolicy,
    ResultsTable,
    paired_difference,
    run_grid,
    run_sequential,
)


class _World:
    def __init__(self, noise: float = 0.0, seed: int = 7) -> None:
        self.noise = noise
        self.seed = seed
        self.calls: list[tuple[float, int]] = []

    def generate(self, config: dict[str, Any], *, rep: int = 0) -> tuple[float, float]:
        self.calls.append((config["x"], rep))
        noise = float(
            np.random.default_rng(self.seed + rep).uniform(-self.noise, self.noise)
        )
        return config["x"], config["x"] + noise


class _Scorer:
    @staticmethod
    def score(truth: float, observations: float, config: dict[str, Any]) -> dict:
        del config
        return {"loss": observations, "reward": 1 - observations, "risk": truth}


@pytest.mark.parametrize("direction", [Direction.MINIMIZE, Direction.MAXIMIZE])
def test_clear_winners_retire_losers_before_maximum(direction: Direction) -> None:
    grid = [{"x": 0.0}, {"x": 1.0}, {"x": 0.5}]
    result = run_sequential(
        _World(),
        _Scorer(),
        grid,
        [Observable("loss", direction)],
        policy=ReplicationPolicy(
            {"loss": (0, 1)}, min_reps=4, max_reps=256, max_evaluations=768
        ),
    )
    best = 0 if direction == Direction.MINIMIZE else 1
    loser = 1 - best
    assert result.stopping_reasons[best] == "selected"
    assert result.stopping_reasons[loser] == "confidently_dominated"
    assert result.allocations[loser] < result.allocations[best] < 256
    assert len(result.results.configs) == sum(result.allocations) < 768
    assert result.summary.metadata[best]["n_reps"] == result.allocations[best]
    assert (
        result.intervals["loss"][best, 0]
        <= grid[best]["x"]
        <= result.intervals["loss"][best, 1]
    )


def test_noisy_near_ties_exhaust_budget_reproducibly_with_raw_rep_ids() -> None:
    grid = [{"x": 0.45}, {"x": 0.46}]
    policy = ReplicationPolicy(
        {"loss": (0, 1)}, min_reps=3, max_reps=100, max_evaluations=31
    )
    first_world, second_world = _World(noise=0.1), _World(noise=0.1)
    args = (_Scorer(), grid, [Observable("loss", Direction.MINIMIZE)])
    first = run_sequential(first_world, *args, policy=policy)
    second = run_sequential(second_world, *args, policy=policy)
    assert first.allocations == second.allocations == [16, 15]
    assert first.stopping_reasons == ["budget_exhausted", "budget_exhausted"]
    assert first_world.calls == second_world.calls
    np.testing.assert_array_equal(first.results.scores, second.results.scores)
    for design, count in enumerate(first.allocations):
        reps = [m["rep"] for m in first.results.metadata if m["design_point"] == design]
        assert reps == list(range(count))
    # Match a common prefix explicitly: allocation may leave unmatched tail reps.
    common = [i for i, m in enumerate(first.results.metadata) if m["rep"] < 15]
    raw = ResultsTable(
        [first.results.configs[i] for i in common],
        first.results.scores[common],
        first.results.observable_names,
        metadata=[first.results.metadata[i] for i in common],
    )
    difference = paired_difference(raw, 0, 1, "loss")
    assert difference.mean == pytest.approx(-0.01)


def test_constraint_boundary_stays_unknown_and_receives_more_replicates() -> None:
    grid = [{"x": 0.1}, {"x": 0.49}]
    policy = ReplicationPolicy(
        {"reward": (0, 1), "risk": (0, 1)},
        min_reps=2,
        max_reps=50,
        max_evaluations=100,
    )
    result = run_sequential(
        _World(),
        _Scorer(),
        grid,
        [
            Observable("reward", Direction.MINIMIZE),
            Observable("risk", Direction.MINIMIZE),
        ],
        policy=policy,
        constraints=[Constraint("safe", "risk", "<=", 0.5)],
    )
    assert result.allocations == [50, 50]
    assert result.feasibility == ["feasible", "unknown"]
    assert result.stopping_reasons == ["max_reps", "max_reps"]


@pytest.mark.parametrize("op", ["<", "<=", ">", ">="])
def test_known_infeasibility_stops_after_initial_replicates(op: str) -> None:
    less = op in {"<", "<="}
    value = 0.9 if less else 0.1
    bounds = (0.8, 1.0) if less else (0.0, 0.2)
    result = run_sequential(
        _World(),
        _Scorer(),
        [{"x": value}],
        [Observable("risk", Direction.MINIMIZE)],
        policy=ReplicationPolicy({"risk": bounds}),
        constraints=[Constraint("safe", "risk", op, 0.5)],
    )
    assert result.allocations == [2]
    assert result.feasibility == ["infeasible"]
    assert result.stopping_reasons == ["confidently_infeasible"]


def test_confident_pareto_tradeoff_and_exact_equivalence() -> None:
    result = run_sequential(
        _World(),
        _Scorer(),
        [{"x": 0.0}, {"x": 1.0}],
        [
            Observable("loss", Direction.MINIMIZE),
            Observable("reward", Direction.MINIMIZE),
        ],
        policy=ReplicationPolicy({"loss": (0, 1), "reward": (0, 1)}, max_reps=100),
    )
    assert result.stopping_reasons == ["resolved_front", "resolved_front"]
    tied = run_sequential(
        _World(),
        _Scorer(),
        [{"x": 0.5}, {"x": 0.5}],
        [Observable("loss", Direction.MINIMIZE)],
        policy=ReplicationPolicy({"loss": (0.5, 0.5)}),
    )
    assert tied.stopping_reasons == ["practically_equivalent", "practically_equivalent"]
    assert tied.allocations == [2, 2]
    np.testing.assert_array_equal(tied.intervals["loss"], np.full((2, 2), 0.5))


def test_raw_unit_equivalence_is_explicit() -> None:
    result = run_sequential(
        _World(),
        _Scorer(),
        [{"x": 0.45}, {"x": 0.46}],
        [Observable("loss", Direction.MINIMIZE)],
        policy=ReplicationPolicy({"loss": (0.4, 0.6)}, equivalence={"loss": 0.2}),
    )
    assert result.allocations == [2, 2]
    assert result.stopping_reasons == [
        "practically_equivalent",
        "practically_equivalent",
    ]


@pytest.mark.parametrize(
    "policy",
    [
        ReplicationPolicy({"loss": (0, 1)}, min_reps=0),
        ReplicationPolicy({"loss": (0, 1)}, min_reps=3, max_reps=2),
        ReplicationPolicy({"loss": (0, 1)}, max_evaluations=1),
        ReplicationPolicy({"loss": (0, 1)}, confidence=1),
        ReplicationPolicy({"missing": (0, 1)}),
        ReplicationPolicy({"loss": (1, 0)}),
        ReplicationPolicy({"loss": (0, float("inf"))}),
        ReplicationPolicy({"loss": (0, 1)}, equivalence={"loss": -1}),
        ReplicationPolicy({"loss": (0, 1)}, equivalence={"unknown": 1}),
    ],
)
def test_invalid_policy_is_rejected_before_evaluation(
    policy: ReplicationPolicy,
) -> None:
    world = _World()
    with pytest.raises(
        ValueError,
        match=r"min_reps|max_evaluations|confidence|bounds|equivalence",
    ):
        run_sequential(
            world,
            _Scorer(),
            [{"x": 0.5}],
            [Observable("loss", Direction.MINIMIZE)],
            policy=policy,
        )
    assert world.calls == []


def test_invalid_constraints_and_out_of_bounds_scores_are_rejected() -> None:
    observables = [Observable("loss", Direction.MINIMIZE)]
    policy = ReplicationPolicy({"loss": (0, 1)})
    for constraint in (
        Constraint("equal", "loss", "==", 0.5),
        Constraint("missing", "unknown", "<=", 0.5),
        Constraint("confidence", "loss", "<=", 0.5, confidence=0.99),
    ):
        with pytest.raises(ValueError, match="Constraints require"):
            run_sequential(
                _World(),
                _Scorer(),
                [{"x": 0.5}],
                observables,
                policy=policy,
                constraints=[constraint],
            )
    with pytest.raises(ValueError, match="violates its declared bounds"):
        run_sequential(_World(), _Scorer(), [{"x": 1.1}], observables, policy=policy)
    with pytest.raises(ValueError, match="at least one design"):
        run_sequential(_World(), _Scorer(), [], observables, policy=policy)
    # Existing fixed replication does not require bounds or change semantics.
    fixed = run_grid(_World(), _Scorer(), [{"x": 1.1}], observables, n_reps=3)
    np.testing.assert_allclose(fixed.scores, 1.1)
