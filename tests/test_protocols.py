"""Tests for Observable, Tier, Direction, and Annotation (issues #1, #3)."""

from __future__ import annotations

import pytest

from trade_study import Direction, Observable, Tier
from trade_study.protocols import Annotation

# -- Tier enum -----------------------------------------------------------------


def test_tier_members() -> None:
    assert set(Tier) == {
        Tier.EMBEDDED,
        Tier.PENALIZED,
        Tier.DIAGNOSTIC,
        Tier.COST,
    }


def test_tier_values() -> None:
    assert Tier.EMBEDDED.value == "embedded"
    assert Tier.PENALIZED.value == "penalized"
    assert Tier.DIAGNOSTIC.value == "diagnostic"
    assert Tier.COST.value == "cost"


def test_tier_lookup_by_value() -> None:
    assert Tier("embedded") is Tier.EMBEDDED


def test_tier_invalid_value_raises() -> None:
    with pytest.raises(ValueError, match="not_a_tier"):
        Tier("not_a_tier")


def test_tier_identity() -> None:
    assert Tier.EMBEDDED is Tier.EMBEDDED


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
        tier=Tier.EMBEDDED,
        direction=Direction.MAXIMIZE,
    )


def test_observable_field_access(obs: Observable) -> None:
    assert obs.name == "coverage_95"
    assert obs.tier is Tier.EMBEDDED
    assert obs.direction is Direction.MAXIMIZE


def test_observable_frozen_prevents_mutation(obs: Observable) -> None:
    with pytest.raises(AttributeError):
        obs.name = "other"  # type: ignore[misc]


def test_observable_equality() -> None:
    a = Observable("rmse", Tier.PENALIZED, Direction.MINIMIZE)
    b = Observable("rmse", Tier.PENALIZED, Direction.MINIMIZE)
    assert a == b


def test_observable_inequality_different_name() -> None:
    a = Observable("rmse", Tier.PENALIZED, Direction.MINIMIZE)
    b = Observable("mae", Tier.PENALIZED, Direction.MINIMIZE)
    assert a != b


def test_observable_inequality_different_tier() -> None:
    a = Observable("rmse", Tier.PENALIZED, Direction.MINIMIZE)
    b = Observable("rmse", Tier.DIAGNOSTIC, Direction.MINIMIZE)
    assert a != b


def test_observable_inequality_different_direction() -> None:
    a = Observable("rmse", Tier.PENALIZED, Direction.MINIMIZE)
    b = Observable("rmse", Tier.PENALIZED, Direction.MAXIMIZE)
    assert a != b


def test_observable_hashable_in_set() -> None:
    a = Observable("rmse", Tier.PENALIZED, Direction.MINIMIZE)
    b = Observable("rmse", Tier.PENALIZED, Direction.MINIMIZE)
    c = Observable("mae", Tier.PENALIZED, Direction.MINIMIZE)
    s = {a, b, c}
    assert len(s) == 2


def test_observable_usable_as_dict_key() -> None:
    obs = Observable("wis", Tier.EMBEDDED, Direction.MINIMIZE)
    d = {obs: 42}
    assert d[Observable("wis", Tier.EMBEDDED, Direction.MINIMIZE)] == 42


def test_observable_repr(obs: Observable) -> None:
    r = repr(obs)
    assert "coverage_95" in r
    assert "Tier.EMBEDDED" in r
    assert "Direction.MAXIMIZE" in r


def test_observable_duplicate_names_different_tiers() -> None:
    a = Observable("x", Tier.EMBEDDED, Direction.MINIMIZE)
    b = Observable("x", Tier.COST, Direction.MINIMIZE)
    assert a != b
    assert len({a, b}) == 2


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
