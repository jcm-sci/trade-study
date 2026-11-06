"""Tests for Monte Carlo power curves and minimum adequate sizes (#231)."""

from __future__ import annotations

from math import ceil, log

import numpy as np
import pytest

from trade_study import (
    ResultsTable,
    minimum_adequate,
    power_curve,
    proportion_interval,
)


def _table(
    success: dict[tuple[str, int], list[int]],
    *,
    extra: dict[str, object] | None = None,
) -> ResultsTable:
    configs: list[dict[str, object]] = []
    scores: list[list[float]] = []
    metadata: list[dict[str, object]] = []
    for point, ((arm, n), outcomes) in enumerate(success.items()):
        for rep, outcome in enumerate(outcomes):
            configs.append({"arm": arm, "n": n, **(extra or {})})
            scores.append([float(outcome)])
            metadata.append({"design_point": point, "rep": rep})
    return ResultsTable(configs, np.array(scores), ["reject"], metadata=metadata)


def test_curves_are_stratified_sorted_and_counted() -> None:
    table = _table({
        ("a", 30): [1, 1, 1, 1],
        ("a", 10): [1, 0, 0, 0],
        ("b", 10): [0, 0, 0, 0],
        ("a", 20): [1, 1, 0, 0],
    })
    curves = power_curve(table, "reject", "n")
    assert [c.stratum for c in curves] == [{"arm": "a"}, {"arm": "b"}]
    a = curves[0]
    assert a.values == (10, 20, 30)
    assert [i.k for i in a.intervals] == [1, 2, 4]
    assert a.intervals[1] == proportion_interval(2, 4)
    assert a.adjustment is None
    assert a.confidence == pytest.approx(0.95)
    assert a.family_confidence == pytest.approx(0.95)


def test_bonferroni_widens_each_interval_by_stratum_size() -> None:
    table = _table({("a", 10): [1, 0, 1, 1], ("a", 20): [1, 1, 1, 0]})
    (curve,) = power_curve(table, "reject", "n", adjustment="bonferroni")
    assert curve.confidence == pytest.approx(0.975)
    assert curve.family_confidence == pytest.approx(0.95)
    assert curve.intervals[0] == proportion_interval(3, 4, confidence=0.975)


def test_minimum_adequate_rules_and_monotonicity() -> None:
    table = _table({
        ("a", 10): [1] * 9 + [0],
        ("a", 20): [1] * 6 + [0] * 4,
        ("a", 30): [1] * 10,
    })
    curves = power_curve(table, "reject", "n")
    (by_estimate,) = minimum_adequate(curves, 0.8, rule="estimate")
    assert by_estimate.value == 10
    assert by_estimate.interval is not None
    assert by_estimate.interval.k == 9
    assert by_estimate.monotone_above is False
    (by_bound,) = minimum_adequate(curves, 0.7)
    assert by_bound.value == 30
    assert by_bound.monotone_above is True
    (none,) = minimum_adequate(curves, 0.99)
    assert none.value is None
    assert none.interval is None
    assert none.monotone_above is False


def test_selection_recovers_analytic_crossing() -> None:
    """Bernoulli power p(n) = 1 - exp(-n / 50); 80% is first reached at 81.

    Scanned sizes sit several Monte Carlo standard errors from 0.8.
    """
    rng = np.random.default_rng(231)
    sizes = [50, 65, 85, 100]
    reps = 4000
    success = {
        ("a", n): rng.binomial(1, 1 - np.exp(-n / 50), reps).tolist() for n in sizes
    }
    (curve,) = power_curve(_table(success), "reject", "n")
    (selected,) = minimum_adequate([curve], 0.8, rule="estimate")
    exact = ceil(-50 * log(0.2))
    assert exact == 81
    assert selected.value == min(n for n in sizes if n >= exact)
    assert selected.monotone_above


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"adjustment": "holm"}, "adjustment"),
        ({"along": "missing"}, "absent"),
    ],
)
def test_power_curve_rejects_invalid_requests(
    kwargs: dict[str, str], message: str
) -> None:
    table = _table({("a", 10): [1, 0]})
    arguments = {"along": "n", **kwargs}
    with pytest.raises(ValueError, match=message):
        power_curve(table, "reject", **arguments)  # type: ignore[arg-type]


def test_power_curve_rejects_non_indicator_and_repeated_values() -> None:
    table = _table({("a", 10): [1, 0]})
    table.scores[0, 0] = 0.5
    with pytest.raises(ValueError, match="0/1"):
        power_curve(table, "reject", "n")
    repeated = _table({("a", 10): [1, 0], ("b", 10): [1, 1]})
    repeated.configs[2]["arm"] = repeated.configs[3]["arm"] = "a"
    with pytest.raises(ValueError, match="repeats"):
        power_curve(repeated, "reject", "n")


def test_power_curve_rejects_unorderable_values() -> None:
    table = _table({("a", 10): [1, 0], ("a", 20): [1, 1]})
    table.configs[2]["n"] = "twenty"
    table.configs[3]["n"] = "twenty"
    with pytest.raises(ValueError, match="orderable"):
        power_curve(table, "reject", "n")


@pytest.mark.parametrize(
    ("target", "rule"), [(1.5, "estimate"), (0.5, "median"), (-0.1, "lower_bound")]
)
def test_minimum_adequate_rejects_invalid_target_or_rule(
    target: float, rule: str
) -> None:
    curves = power_curve(_table({("a", 10): [1, 0]}), "reject", "n")
    with pytest.raises(ValueError, match=r"target|rule"):
        minimum_adequate(curves, target, rule=rule)  # type: ignore[arg-type]
