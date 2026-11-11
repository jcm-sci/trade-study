"""Tests for exact event-driven trial design calculations (#233)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from trade_study import (
    AccrualPlan,
    alpha_spending,
    conditional_power,
    critical_count,
    design_group_sequential,
    efficacy_from_share,
    efficacy_interval,
    efficacy_share,
    events_required,
    expected_events_by_period,
    max_events_for_power,
    operating_characteristics,
    periods_to_events,
    unconditional_power,
)


@pytest.mark.parametrize("efficacy", [-0.5, 0.0, 0.3, 0.9])
@pytest.mark.parametrize("rho", [0.5, 1.0, 2.0])
def test_share_round_trip(efficacy: float, rho: float) -> None:
    share = efficacy_share(efficacy, rho)
    assert efficacy_from_share(share, rho) == pytest.approx(efficacy)


def test_interval_matches_clopper_pearson() -> None:
    result = efficacy_interval(5, 40, margin=0.3, rho=1.5)
    hi = stats.beta.ppf(0.975, 6, 35)
    lo = stats.beta.ppf(0.025, 5, 36)
    assert result.lower == pytest.approx(efficacy_from_share(hi, 1.5))
    assert result.upper == pytest.approx(efficacy_from_share(lo, 1.5))
    assert result.success


def test_interval_edges_and_validation() -> None:
    empty = efficacy_interval(0, 0)
    assert math.isnan(empty.estimate)
    assert not empty.success
    assert efficacy_interval(3, 3).lower == -math.inf
    assert efficacy_interval(0, 10).upper == pytest.approx(1.0)
    with pytest.raises(ValueError, match="treatment_events"):
        efficacy_interval(4, 3)
    with pytest.raises(ValueError, match="confidence"):
        efficacy_interval(1, 3, confidence=1.0)


def test_events_required_is_a_stable_threshold() -> None:
    target = events_required(0.7, margin=0.3, max_events=300)
    assert all(conditional_power(d, 0.7, margin=0.3) >= 0.9 for d in range(target, 301))
    assert conditional_power(target - 1, 0.7, margin=0.3) < 0.9
    assert events_required(0.7, power=0.0) == 0
    with pytest.raises(ValueError, match="No event target"):
        events_required(0.35, power=0.99, margin=0.3, max_events=40)


def test_unconditional_power_increases() -> None:
    assert (
        0
        < unconditional_power(10, 0.7, margin=0.3)
        < unconditional_power(60, 0.7, margin=0.3)
    )
    assert conditional_power(1, 0.7, margin=0.3) == pytest.approx(0.0)


def test_spending_functions() -> None:
    assert alpha_spending(1.0, 0.025, "pocock") == pytest.approx(0.025)
    assert alpha_spending(0.25, 0.025) < alpha_spending(0.5, 0.025) < 0.025
    with pytest.raises(ValueError, match="kind"):
        alpha_spending(0.5, 0.025, "haybittle")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="fraction"):
        alpha_spending(0.0, 0.025)


def test_single_look_equals_fixed_design() -> None:
    design = design_group_sequential([78], margin=0.3)
    oc = operating_characteristics(design, 0.7)
    assert oc.power == pytest.approx(conditional_power(78, 0.7, margin=0.3))
    assert oc.expected_events == pytest.approx(78)


@pytest.mark.parametrize("spending", ["obf", "pocock"])
def test_type_one_error_is_controlled(spending: str) -> None:
    design = design_group_sequential(
        [30, 60, 90, 120],
        margin=0.3,
        spending=spending,  # type: ignore[arg-type]
    )
    oc = operating_characteristics(design, 0.3, futility_threshold=None)
    assert oc.power <= 0.025 + 1e-12
    assert oc.power == pytest.approx(design.alpha_spent[-1])
    assert all(
        b >= a for a, b in zip(design.alpha_spent, design.alpha_spent[1:], strict=False)
    )


def test_obf_spends_less_early_than_pocock() -> None:
    obf = design_group_sequential([40, 80], margin=0.3)
    pocock = design_group_sequential([40, 80], margin=0.3, spending="pocock")
    assert obf.alpha_spent[0] < pocock.alpha_spent[0]
    assert obf.critical[0] <= pocock.critical[0]


def test_exact_dynamic_program_matches_simulation() -> None:
    design = design_group_sequential([30, 60, 90], margin=0.3, spending="pocock")
    oc = operating_characteristics(design, 0.7, futility_threshold=0.0)
    rng = np.random.default_rng(233)
    theta = efficacy_share(0.7)
    draws = rng.binomial(1, theta, (40_000, 90)).cumsum(axis=1)
    alive = np.ones(draws.shape[0], dtype=bool)
    efficacy_stops, futility_stops = [], []
    for j, (look, c) in enumerate(zip(design.looks, design.critical, strict=True)):
        k = draws[:, look - 1]
        stop = alive & (k <= c)
        efficacy_stops.append(stop.mean())
        alive &= ~stop
        if j < len(design.looks) - 1:
            futile = alive & (k >= math.ceil(look / 2))
            futility_stops.append(futile.mean())
            alive &= ~futile
    assert np.allclose(oc.efficacy_stop, efficacy_stops, atol=0.01)
    assert np.allclose(oc.futility_stop[:-1], futility_stops, atol=0.01)
    assert oc.futility_stop[-1] == pytest.approx(0.0)


def test_group_sequential_validation_and_planning() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        design_group_sequential([40, 40])
    design = max_events_for_power([0.5, 1.0], 0.7, margin=0.3)
    assert operating_characteristics(design, 0.7).power >= 0.9
    with pytest.raises(ValueError, match="No maximum event total"):
        max_events_for_power([1.0], 0.31, margin=0.3, max_events=60)


def test_accrual_closed_form_and_ramp() -> None:
    flat = expected_events_by_period(
        AccrualPlan(enrolled=1000), control_rate=0.01, efficacy=0.5, periods=3
    )
    assert flat == pytest.approx([7.5, 15.0, 22.5])
    dropout = AccrualPlan(enrolled=1000, dropout_per_period=0.1)
    decayed = expected_events_by_period(
        dropout, control_rate=0.01, efficacy=0.0, periods=1, substeps=400
    )
    assert decayed[0] == pytest.approx(
        1000 * 0.01 * (1 - math.exp(-0.1)) / 0.1, rel=1e-4
    )
    ramp = expected_events_by_period(
        AccrualPlan(enrolled=1000, ramp_periods=2.0, seasonal_amplitude=0.5),
        control_rate=0.01,
        efficacy=0.0,
        periods=24,
    )
    assert np.all(np.diff(ramp) > 0)
    assert periods_to_events(ramp, ramp[5]) == pytest.approx(6.0)
    assert periods_to_events(ramp, 1e9) == math.inf


def test_critical_count_is_the_success_boundary() -> None:
    for total in (10, 40, 78):
        k = critical_count(total, margin=0.3)
        assert k >= 0
        assert efficacy_interval(k, total, margin=0.3).success
        assert not efficacy_interval(k + 1, total, margin=0.3).success
        theta = efficacy_share(0.7, 1.0)
        power = float(stats.binom.cdf(k, total, theta))
        assert power == pytest.approx(conditional_power(total, 0.7, margin=0.3))
    assert critical_count(2, margin=0.3) == -1
    assert critical_count(0) == -1


def test_critical_count_validates_arguments() -> None:
    with pytest.raises(ValueError, match="total_events"):
        critical_count(-1)
    with pytest.raises(ValueError, match="total_events"):
        critical_count(total_events=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="total_events"):
        critical_count(3.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="confidence"):
        critical_count(10, confidence=1.0)
