"""Tests for protocol types (issues #1, #2, #3, #4)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from trade_study import Direction, Observable
from trade_study.protocols import Annotation, ResultsTable, Scorer, Simulator

# -- Direction enum ------------------------------------------------------------


def test_direction_members() -> None:
    assert set(Direction) == {Direction.MINIMIZE, Direction.MAXIMIZE}


def test_direction_values() -> None:
    assert Direction.MINIMIZE.value == "minimize"
    assert Direction.MAXIMIZE.value == "maximize"


def test_direction_lookup_by_value() -> None:
    assert Direction("maximize") is Direction.MAXIMIZE


def test_direction_invalid_value_raises() -> None:
    with pytest.raises(ValueError, match="bad"):
        Direction("bad")


# -- Observable frozen dataclass -----------------------------------------------


@pytest.fixture
def obs() -> Observable:
    return Observable(
        name="coverage_95",
        direction=Direction.MAXIMIZE,
    )


def test_observable_field_access(obs: Observable) -> None:
    assert obs.name == "coverage_95"
    assert obs.direction is Direction.MAXIMIZE


def test_observable_frozen_prevents_mutation(obs: Observable) -> None:
    with pytest.raises(AttributeError):
        obs.name = "other"  # type: ignore[misc]


def test_observable_equality() -> None:
    a = Observable("rmse", Direction.MINIMIZE)
    b = Observable("rmse", Direction.MINIMIZE)
    assert a == b


def test_observable_inequality_different_name() -> None:
    a = Observable("rmse", Direction.MINIMIZE)
    b = Observable("mae", Direction.MINIMIZE)
    assert a != b


def test_observable_inequality_different_direction() -> None:
    a = Observable("rmse", Direction.MINIMIZE)
    b = Observable("rmse", Direction.MAXIMIZE)
    assert a != b


def test_observable_hashable_in_set() -> None:
    a = Observable("rmse", Direction.MINIMIZE)
    b = Observable("rmse", Direction.MINIMIZE)
    c = Observable("mae", Direction.MINIMIZE)
    s = {a, b, c}
    assert len(s) == 2


def test_observable_usable_as_dict_key() -> None:
    obs = Observable("wis", Direction.MINIMIZE)
    d = {obs: 42}
    assert d[Observable("wis", Direction.MINIMIZE)] == 42


def test_observable_repr(obs: Observable) -> None:
    r = repr(obs)
    assert "coverage_95" in r
    assert "Direction.MAXIMIZE" in r


# -- Annotation.resolve -------------------------------------------------------


@pytest.fixture
def cost_lookup() -> dict[str, float]:
    return {"low": 10.0, "medium": 50.0, "high": 100.0}


def test_annotation_resolve_dict_lookup(cost_lookup: dict[str, float]) -> None:
    ann = Annotation(name="cost", lookup=cost_lookup, key="budget")
    result = ann.resolve({"budget": "medium"})
    assert result == pytest.approx(50.0)


def test_annotation_resolve_dict_all_keys(cost_lookup: dict[str, float]) -> None:
    ann = Annotation(name="cost", lookup=cost_lookup, key="level")
    for key, expected in cost_lookup.items():
        assert ann.resolve({"level": key}) == expected


def test_annotation_resolve_callable() -> None:
    ann = Annotation(name="scaled", lookup=lambda x: x * 2.5, key="n")
    assert ann.resolve({"n": 4}) == pytest.approx(10.0)


def test_annotation_resolve_callable_returns_float() -> None:
    ann = Annotation(name="flag", lookup=lambda x: x, key="val")
    result = ann.resolve({"val": 1})
    assert isinstance(result, float)


def test_annotation_resolve_missing_config_key() -> None:
    ann = Annotation(name="cost", lookup={"a": 1.0}, key="budget")
    with pytest.raises(KeyError, match="budget"):
        ann.resolve({"other_key": "a"})


def test_annotation_resolve_missing_lookup_key() -> None:
    ann = Annotation(name="cost", lookup={"a": 1.0}, key="budget")
    with pytest.raises(KeyError, match="missing"):
        ann.resolve({"budget": "missing"})


# -- Simulator protocol --------------------------------------------------------


class _ToySimulator:
    def generate(self, config: dict[str, Any]) -> tuple[Any, Any]:
        return config["mu"], config["mu"] + 0.1


class _BadSimulator:
    def not_generate(self, config: dict[str, Any]) -> tuple[Any, Any]:
        return 0.0, 0.0


def test_simulator_isinstance_conforming() -> None:
    assert isinstance(_ToySimulator(), Simulator)


def test_simulator_isinstance_non_conforming() -> None:
    assert not isinstance(_BadSimulator(), Simulator)


def test_simulator_generate_returns_tuple() -> None:
    sim = _ToySimulator()
    result = sim.generate({"mu": 5.0})
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_simulator_generate_uses_config() -> None:
    sim = _ToySimulator()
    truth, obs = sim.generate({"mu": 3.0})
    assert truth == pytest.approx(3.0)
    assert obs == pytest.approx(3.1)


# -- Scorer protocol -----------------------------------------------------------


class _ToyScorer:
    def score(
        self,
        truth: Any,
        observations: Any,
        config: dict[str, Any],
    ) -> dict[str, float]:
        return {"error": abs(truth - observations)}


class _BadScorer:
    def evaluate(self, truth: Any) -> float:
        return 0.0


def test_scorer_isinstance_conforming() -> None:
    assert isinstance(_ToyScorer(), Scorer)


def test_scorer_isinstance_non_conforming() -> None:
    assert not isinstance(_BadScorer(), Scorer)


def test_scorer_returns_dict() -> None:
    s = _ToyScorer()
    result = s.score(5.0, 5.3, {})
    assert isinstance(result, dict)
    assert "error" in result


def test_scorer_values_are_float() -> None:
    s = _ToyScorer()
    result = s.score(5.0, 5.3, {})
    assert all(isinstance(v, float) for v in result.values())


def test_simulator_and_scorer_end_to_end() -> None:
    sim = _ToySimulator()
    scorer = _ToyScorer()
    truth, obs = sim.generate({"mu": 2.0})
    scores = scorer.score(truth, obs, {"mu": 2.0})
    assert scores["error"] == pytest.approx(0.1)


# -- ResultsTable construction -------------------------------------------------


def test_results_table_scores_shape() -> None:
    rt = ResultsTable(
        configs=[{"a": 1}, {"a": 2}, {"a": 3}],
        scores=np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
        observable_names=["rmse", "coverage"],
    )
    assert rt.scores.shape == (3, 2)
    assert rt.scores.shape == (len(rt.configs), len(rt.observable_names))


def test_results_table_annotations_shape() -> None:
    rt = ResultsTable(
        configs=[{"a": 1}, {"a": 2}],
        scores=np.array([[0.1], [0.2]]),
        observable_names=["rmse"],
        annotations=np.array([[10.0, 20.0], [30.0, 40.0]]),
        annotation_names=["cost", "time"],
    )
    assert rt.annotations is not None
    assert rt.annotations.shape == (2, 2)
    assert rt.annotations.shape == (len(rt.configs), len(rt.annotation_names))


def test_results_table_annotations_default_none() -> None:
    rt = ResultsTable(
        configs=[{"a": 1}],
        scores=np.array([[0.5]]),
        observable_names=["rmse"],
    )
    assert rt.annotations is None
    assert rt.annotation_names == []


def test_results_table_metadata_default_empty() -> None:
    rt = ResultsTable(
        configs=[{"a": 1}],
        scores=np.array([[0.5]]),
        observable_names=["rmse"],
    )
    assert rt.metadata == []


def test_results_table_metadata_alignment() -> None:
    rt = ResultsTable(
        configs=[{"a": 1}, {"a": 2}],
        scores=np.array([[0.1], [0.2]]),
        observable_names=["rmse"],
        metadata=[{"wall": 0.1}, {"wall": 0.2}],
    )
    assert len(rt.metadata) == len(rt.configs)


def test_results_table_single_trial() -> None:
    rt = ResultsTable(
        configs=[{"x": 42}],
        scores=np.array([[1.0, 2.0, 3.0]]),
        observable_names=["a", "b", "c"],
    )
    assert rt.scores.shape == (1, 3)
    assert len(rt.configs) == 1


def test_results_table_empty() -> None:
    rt = ResultsTable(
        configs=[],
        scores=np.empty((0, 2)),
        observable_names=["rmse", "coverage"],
    )
    assert rt.scores.shape == (0, 2)
    assert len(rt.configs) == 0


def test_results_table_observable_names_order() -> None:
    names = ["coverage", "rmse", "wall_time"]
    rt = ResultsTable(
        configs=[{"a": 1}],
        scores=np.array([[0.95, 0.1, 3.2]]),
        observable_names=names,
    )
    assert rt.observable_names == ["coverage", "rmse", "wall_time"]
