"""Budgeted replication under near ties, feasibility boundaries, and clear gaps."""

from __future__ import annotations

import json
from dataclasses import replace
from math import log, sqrt
from typing import TYPE_CHECKING, Any

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
from trade_study.artifacts import load_result_bundle, save_result_bundle

if TYPE_CHECKING:
    from pathlib import Path


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


def test_fixed_plan_completes_before_its_single_look() -> None:
    """A clear separation cannot retire a fixed-plan design before its declared n."""
    world = _World()
    result = run_sequential(
        world,
        _Scorer(),
        [{"x": 0.0}, {"x": 1.0}],
        [Observable("loss", Direction.MINIMIZE)],
        policy=ReplicationPolicy(
            {"loss": (0, 1)},
            min_reps=2,
            max_reps=7,
            max_evaluations=14,
            allocation="fixed",
        ),
    )
    assert result.allocations == [7, 7]
    assert result.stopping_reasons == ["fixed_plan_complete"] * 2
    assert world.calls == [(x, rep) for rep in range(7) for x in (0.0, 1.0)]
    assert all(meta["allocation"] == "fixed" for meta in result.results.metadata)
    radius = sqrt(log(2 * 2 / 0.05) / (2 * 7))
    np.testing.assert_allclose(result.intervals["loss"], [[0, radius], [1 - radius, 1]])
    for row, meta in enumerate(result.summary.metadata):
        contract = meta["replication_contract"]
        assert contract["planned_reps"] == contract["actual_reps"] == 7
        assert contract["actual_evaluations"] == 14
        assert contract["max_looks_per_stream"] == 1
        assert contract["multiplicity"] == 2
        assert contract["decision_status"] == "not_assessed"
        assert contract["interval_half_widths"]["loss"] == pytest.approx(
            (result.intervals["loss"][row, 1] - result.intervals["loss"][row, 0]) / 2
        )


def test_adaptive_contract_keeps_power_and_stopping_distinct(tmp_path: Path) -> None:
    """A budget-limited CRN comparison exports its real denominator and assumptions."""
    observables = [Observable("loss", Direction.MINIMIZE)]
    result = run_sequential(
        _World(noise=0.01),
        _Scorer(),
        [{"x": 0.45}, {"x": 0.46}],
        observables,
        policy=ReplicationPolicy({"loss": (0, 1)}, max_reps=17, max_evaluations=34),
    )
    assert result.allocations == [17, 17]
    for meta in result.summary.metadata:
        contract = meta["replication_contract"]
        assert contract["planned_reps"] is None
        assert contract["actual_reps"] == meta["n_reps"] == 17
        assert contract["max_looks_per_stream"] == 17
        assert contract["multiplicity"] == 34
        assert contract["per_look_error"] == pytest.approx(0.05 / 34)
        assert contract["decision_status"] == "unresolved"
        assert contract["gate_power"] == {"status": "not_assessed", "value": None}
        assert contract["gate_false_failure_rate"] == contract["gate_power"]
        assert contract["paired_contrast_coverage"] == "not_supplied"
        assert contract["failure_policy"] == "abort"
        assert contract["failed_reps"] == 0
        json.dumps(contract, allow_nan=False)
    bundle = tmp_path / "summary"
    save_result_bundle(result.summary, bundle, observables=observables)
    restored = load_result_bundle(bundle).results
    assert restored.metadata == result.summary.metadata
    np.testing.assert_array_equal(restored.scores, result.summary.scores)


def test_used_policy_is_detached_from_callback_mutation() -> None:
    """Caller-owned dictionary changes cannot narrow bounds after seeing results."""
    bounds = {"loss": (0.0, 1.0)}
    tolerances = {"loss": 0.0}

    class _MutatingWorld(_World):
        def generate(
            self, config: dict[str, Any], *, rep: int = 0
        ) -> tuple[float, float]:
            outcome = super().generate(config, rep=rep)
            bounds["loss"] = (0.5, 0.5)
            tolerances["loss"] = 1.0
            return outcome

    result = run_sequential(
        _MutatingWorld(),
        _Scorer(),
        [{"x": 0.2}, {"x": 0.8}],
        [Observable("loss", Direction.MINIMIZE)],
        policy=ReplicationPolicy(bounds, max_reps=3, equivalence=tolerances),
    )
    assert result.policy.score_bounds == {"loss": (0, 1)}
    assert result.policy.equivalence == {"loss": 0}
    contract = result.summary.metadata[0]["replication_contract"]
    assert contract["score_bounds"] == {"loss": [0, 1]}
    assert contract["equivalence"] == {"loss": 0}
    result.policy.score_bounds["loss"] = (-100, 100)
    assert contract["score_bounds"] == {"loss": [0, 1]}


@pytest.mark.parametrize("field_name", ["min_reps", "max_reps", "max_evaluations"])
@pytest.mark.parametrize("value", [True, 2.0, 1.5])
def test_noninteger_budgets_fail_before_simulator_calls(
    field_name: str, value: object
) -> None:
    """A malformed horizon cannot alter the inference family after evaluation starts."""
    world = _World()
    policy = replace(ReplicationPolicy({"loss": (0, 1)}), **{field_name: value})
    with pytest.raises(ValueError, match="integers"):
        run_sequential(
            world,
            _Scorer(),
            [{"x": 0.5}],
            [Observable("loss", Direction.MINIMIZE)],
            policy=policy,
        )
    assert not world.calls


def test_incomplete_fixed_budget_is_refused_before_evaluation() -> None:
    """A fixed plan cannot silently degrade into a result-dependent partial plan."""
    world = _World()
    policy = ReplicationPolicy(
        {"loss": (0, 1)}, max_reps=7, max_evaluations=5, allocation="fixed"
    )
    with pytest.raises(ValueError, match="Fixed allocation"):
        run_sequential(
            world,
            _Scorer(),
            [{"x": 0.2}, {"x": 0.8}],
            [Observable("loss", Direction.MINIMIZE)],
            policy=policy,
        )
    assert not world.calls


@pytest.mark.parametrize(
    ("field_name", "value"),
    [("allocation", "unknown"), ("replicate_unit", "dependent_rows")],
)
def test_unsupported_assumptions_are_not_silently_reused(
    field_name: str, value: str
) -> None:
    """Unsupported dependence/policies cannot acquire independent coverage."""
    world = _World()
    policy = replace(ReplicationPolicy({"loss": (0, 1)}), **{field_name: value})
    with pytest.raises(ValueError, match="Unsupported"):
        run_sequential(
            world,
            _Scorer(),
            [{"x": 0.5}],
            [Observable("loss", Direction.MINIMIZE)],
            policy=policy,
        )
    assert not world.calls


def test_numpy_integer_budgets_produce_primitive_metadata() -> None:
    """NumPy counts are accepted as integers and normalized for artifact export."""
    result = run_sequential(
        _World(),
        _Scorer(),
        [{"x": 0.5}],
        [Observable("loss", Direction.MINIMIZE)],
        policy=ReplicationPolicy(
            {"loss": (0, 1)},
            min_reps=np.int64(2),
            max_reps=np.int64(3),
            max_evaluations=np.int64(3),
            allocation="fixed",
        ),
    )
    contract = result.summary.metadata[0]["replication_contract"]
    assert type(contract["actual_reps"]) is int
    assert type(result.policy.max_reps) is int
    json.dumps(contract, allow_nan=False)


def test_independent_cluster_scores_keep_the_cluster_denominator() -> None:
    """Dependent observations inside one cluster never become extra replicates."""

    class _ClusterWorld:
        @staticmethod
        def generate(
            config: dict[str, Any], *, rep: int = 0
        ) -> tuple[float, np.ndarray]:
            draw = np.random.default_rng(20 + rep).binomial(1, config["p"])
            return config["p"], np.full(10, draw)

    class _ClusterScorer:
        @staticmethod
        def score(
            truth: float, observations: np.ndarray, config: dict[str, Any]
        ) -> dict[str, float]:
            del truth, config
            return {"loss": float(np.mean(observations))}

    result = run_sequential(
        _ClusterWorld(),
        _ClusterScorer(),
        [{"p": 0.5}],
        [Observable("loss", Direction.MINIMIZE)],
        policy=ReplicationPolicy(
            {"loss": (0, 1)},
            max_reps=5,
            max_evaluations=5,
            allocation="fixed",
            replicate_unit="independent_cluster",
        ),
    )
    assert result.allocations == [5]
    contract = result.summary.metadata[0]["replication_contract"]
    assert contract["replicate_unit"] == "independent_cluster"
    assert contract["actual_reps"] == 5
    assert contract["estimand"] == "mean_score_per_declared_replicate"
    assert result.intervals["loss"][0, 0] <= 0.5 <= result.intervals["loss"][0, 1]


@pytest.mark.parametrize("bad_score", [None, np.nan, np.inf, 2.0])
def test_failed_replicate_never_returns_a_reduced_denominator(
    bad_score: float | None,
) -> None:
    """Missing/failed second evaluations abort; a successful first row is not a pass."""

    class _FailingScorer:
        calls = 0

        def score(
            self, truth: float, observations: float, config: dict[str, Any]
        ) -> dict[str, float]:
            del truth, observations, config
            self.calls += 1
            if self.calls == 1:
                return {"loss": 0.5}
            return {} if bad_score is None else {"loss": bad_score}

    world = _World()
    with pytest.raises(ValueError, match="violates its declared bounds"):
        run_sequential(
            world,
            _FailingScorer(),
            [{"x": 0.5}],
            [Observable("loss", Direction.MINIMIZE)],
            policy=ReplicationPolicy({"loss": (0, 1)}, max_reps=4, allocation="fixed"),
        )
    assert world.calls == [(0.5, 0), (0.5, 1)]
