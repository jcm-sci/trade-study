"""Exact design calculations for event-driven two-arm comparative trials.

Treatment and control events are independent Poisson processes with rate
ratio ``1 - efficacy``. Given ``D`` total events and a treatment-to-control
person-time ratio ``rho``, treatment-arm events are ``Binomial(D, theta)``
with ``theta = rho (1 - E) / (rho (1 - E) + 1)``. Conditioning on ``D``
removes the unknown baseline rate, so inference and power are exact
binomial calculations.

Assumptions callers establish: proportional rates over follow-up, a known
person-time ratio, and independent events across participants. Clustered,
misclassified or time-varying settings need simulation, for example with
:func:`~trade_study.run_grid` and :func:`~trade_study.power_curve`.

All functions require SciPy (the ``stats`` extra).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cache
from itertools import pairwise
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from types import ModuleType

    from numpy.typing import NDArray

_SPENDING = ("obf", "pocock")


def _scipy_stats() -> ModuleType:
    try:
        from scipy import stats  # type: ignore[import-untyped]
    except ImportError as error:
        msg = "trade_study.trials requires trade-study[stats] (SciPy >=1.11)"
        raise ImportError(msg) from error
    module: ModuleType = stats
    return module


def efficacy_share(efficacy: float, rho: float = 1.0) -> float:
    """Expected treatment-arm share of events.

    Args:
        efficacy: ``1 - rate ratio``, at most 1.
        rho: Treatment-to-control person-time ratio (positive).

    Returns:
        ``theta = rho (1 - E) / (rho (1 - E) + 1)``.

    Examples:
        >>> efficacy_share(0.0)
        0.5
    """
    ratio = rho * (1.0 - efficacy)
    return ratio / (ratio + 1.0)


def efficacy_from_share(theta: float, rho: float = 1.0) -> float:
    """Invert :func:`efficacy_share`.

    Args:
        theta: Treatment-arm share of events in [0, 1).
        rho: Treatment-to-control person-time ratio.

    Returns:
        The implied efficacy.

    Examples:
        >>> round(efficacy_from_share(efficacy_share(0.7, 2.0), 2.0), 12)
        0.7
    """
    return 1.0 - theta / (rho * (1.0 - theta))


@dataclass(frozen=True)
class EfficacyInterval:
    """Exact conditional efficacy estimate and two-sided interval.

    Attributes:
        treatment_events: Events in the treatment arm.
        total_events: Events in both arms.
        estimate: Conditional maximum-likelihood efficacy (``nan`` without
            events; ``-inf`` when every event is in the treatment arm).
        lower: Lower bound (``-inf`` when undefined).
        upper: Upper bound.
        margin: Success requires ``lower > margin``.
        confidence: Two-sided confidence level.
        success: Whether ``lower > margin``.
    """

    treatment_events: int
    total_events: int
    estimate: float
    lower: float
    upper: float
    margin: float
    confidence: float
    success: bool


def efficacy_interval(
    treatment_events: int,
    total_events: int,
    *,
    rho: float = 1.0,
    margin: float = 0.0,
    confidence: float = 0.95,
) -> EfficacyInterval:
    """Estimate efficacy with a Clopper-Pearson interval on the event share.

    Args:
        treatment_events: Events in the treatment arm.
        total_events: Events in both arms.
        rho: Treatment-to-control person-time ratio.
        margin: Null efficacy margin for the success criterion.
        confidence: Two-sided confidence level in (0, 1).

    Returns:
        Estimate, exact bounds and success indicator.

    Raises:
        ValueError: If counts or levels are invalid.

    Examples:
        >>> result = efficacy_interval(5, 40, margin=0.3)
        >>> round(result.estimate, 3), result.success
        (0.857, True)
    """
    if not 0 <= treatment_events <= total_events:
        msg = "Require 0 <= treatment_events <= total_events"
        raise ValueError(msg)
    if not 0 < confidence < 1 or rho <= 0:
        msg = "confidence must lie in (0, 1) and rho must be positive"
        raise ValueError(msg)
    k, n = treatment_events, total_events
    if n == 0:
        return EfficacyInterval(
            0, 0, math.nan, -math.inf, 1.0, margin, confidence, success=False
        )
    beta = _scipy_stats().beta
    tail = (1.0 - confidence) / 2
    theta_hi = 1.0 if k == n else float(beta.ppf(1 - tail, k + 1, n - k))
    theta_lo = 0.0 if k == 0 else float(beta.ppf(tail, k, n - k + 1))
    lower = -math.inf if theta_hi >= 1.0 else efficacy_from_share(theta_hi, rho)
    estimate = -math.inf if k == n else efficacy_from_share(k / n, rho)
    return EfficacyInterval(
        k,
        n,
        estimate,
        lower,
        efficacy_from_share(theta_lo, rho),
        margin,
        confidence,
        success=lower > margin,
    )


@cache
def _critical(total: int, rho: float, margin: float, confidence: float) -> int:
    """Largest treatment-arm count that succeeds, or -1.

    Returns:
        Critical count.
    """
    critical = -1
    for k in range(total + 1):
        if not efficacy_interval(
            k, total, rho=rho, margin=margin, confidence=confidence
        ).success:
            break
        critical = k
    return critical


def conditional_power(
    total_events: int,
    efficacy: float,
    *,
    rho: float = 1.0,
    margin: float = 0.0,
    confidence: float = 0.95,
) -> float:
    """Exact power of the fixed-sample success rule given ``D`` events.

    Args:
        total_events: Total events ``D``.
        efficacy: True efficacy.
        rho: Treatment-to-control person-time ratio.
        margin: Null efficacy margin.
        confidence: Two-sided confidence level.

    Returns:
        Probability that the lower bound exceeds ``margin``.

    Examples:
        >>> conditional_power(78, 0.7, margin=0.3) >= 0.9
        True
    """
    critical = _critical(total_events, rho, margin, confidence)
    if critical < 0:
        return 0.0
    theta = efficacy_share(efficacy, rho)
    return float(_scipy_stats().binom.cdf(critical, total_events, theta))


def events_required(
    efficacy: float,
    *,
    power: float = 0.9,
    rho: float = 1.0,
    margin: float = 0.0,
    confidence: float = 0.95,
    max_events: int = 2000,
) -> int:
    """Smallest event target whose exact power, and every larger one, meets ``power``.

    Exact power is saw-toothed in ``D``; requiring every larger target up to
    ``max_events`` to qualify avoids selecting an isolated peak.

    Args:
        efficacy: Assumed true efficacy.
        power: Target power in [0, 1].
        rho: Treatment-to-control person-time ratio.
        margin: Null efficacy margin.
        confidence: Two-sided confidence level.
        max_events: Search limit.

    Returns:
        The event target.

    Raises:
        ValueError: If no target up to ``max_events`` qualifies.

    Examples:
        >>> events_required(0.7, margin=0.3)
        78
    """
    powers = np.array([
        conditional_power(d, efficacy, rho=rho, margin=margin, confidence=confidence)
        for d in range(max_events + 1)
    ])
    failing = np.flatnonzero(powers < power)
    if failing.size == 0:
        return 0
    if failing[-1] == max_events:
        msg = f"No event target up to {max_events} reaches power {power}"
        raise ValueError(msg)
    return int(failing[-1] + 1)


def unconditional_power(
    expected_control_events: float,
    efficacy: float,
    *,
    rho: float = 1.0,
    margin: float = 0.0,
    confidence: float = 0.95,
) -> float:
    """Power of a fixed-duration trial with a Poisson event total.

    The total is ``Poisson(mu_c (1 + rho (1 - E)))`` and the analysis
    conditions on its realized value.

    Args:
        expected_control_events: Expected control-arm events.
        efficacy: True efficacy.
        rho: Treatment-to-control person-time ratio.
        margin: Null efficacy margin.
        confidence: Two-sided confidence level.

    Returns:
        Probability of meeting the success criterion.

    Examples:
        >>> unconditional_power(0.0, 0.7)
        0.0
    """
    mean_total = expected_control_events * (1.0 + rho * (1.0 - efficacy))
    if mean_total <= 0:
        return 0.0
    poisson = _scipy_stats().poisson
    totals = np.arange(int(poisson.ppf(1 - 1e-10, mean_total)) + 2)
    weights = poisson.pmf(totals, mean_total)
    conditional = [
        conditional_power(
            int(d), efficacy, rho=rho, margin=margin, confidence=confidence
        )
        for d in totals
    ]
    return float(np.dot(weights, conditional))


def alpha_spending(
    fraction: float, alpha: float, kind: Literal["obf", "pocock"] = "obf"
) -> float:
    """Lan-DeMets cumulative one-sided alpha spent by an information fraction.

    ``"obf"``: ``2 (1 - Phi(z_{1 - alpha/2} / sqrt(t)))``.
    ``"pocock"``: ``alpha log(1 + (e - 1) t)``.

    Args:
        fraction: Information fraction in (0, 1].
        alpha: Total one-sided alpha.
        kind: Spending-function family.

    Returns:
        Cumulative alpha spent.

    Raises:
        ValueError: If ``kind`` or ``fraction`` is invalid.

    Examples:
        >>> round(alpha_spending(1.0, 0.025), 6)
        0.025
        >>> alpha_spending(0.5, 0.025) < alpha_spending(0.5, 0.025, "pocock")
        True
    """
    if kind not in _SPENDING:
        msg = f"kind must be one of {_SPENDING}"
        raise ValueError(msg)
    if not 0 < fraction <= 1:
        msg = "fraction must lie in (0, 1]"
        raise ValueError(msg)
    if kind == "pocock":
        return alpha * math.log(1 + (math.e - 1) * fraction)
    norm = _scipy_stats().norm
    z = norm.ppf(1 - alpha / 2)
    return float(2 * (1 - norm.cdf(z / math.sqrt(fraction))))


@dataclass(frozen=True)
class GroupSequentialDesign:
    """Exact efficacy boundaries for an event-driven trial.

    Attributes:
        looks: Cumulative event totals at each analysis.
        critical: Largest cumulative treatment-arm count that stops for
            efficacy at each look (``-1`` when none can).
        alpha_spent: Cumulative one-sided alpha actually spent at each look.
        margin: Null efficacy margin.
        alpha: Nominal total one-sided alpha.
        rho: Treatment-to-control person-time ratio.
        spending: Spending-function family.
    """

    looks: tuple[int, ...]
    critical: tuple[int, ...]
    alpha_spent: tuple[float, ...]
    margin: float
    alpha: float
    rho: float
    spending: str


def _convolve(
    dist: NDArray[np.float64], new_events: int, theta: float
) -> NDArray[np.float64]:
    increment = _scipy_stats().binom.pmf(np.arange(new_events + 1), new_events, theta)
    return np.convolve(dist, increment)


def design_group_sequential(
    looks: list[int],
    *,
    margin: float = 0.0,
    alpha: float = 0.025,
    rho: float = 1.0,
    spending: Literal["obf", "pocock"] = "obf",
) -> GroupSequentialDesign:
    """Compute exact efficacy boundaries under Lan-DeMets alpha spending.

    At each look the critical count is the largest whose exact null
    probability of first crossing does not exceed the alpha newly available.
    Discreteness means the alpha actually spent can be below the nominal
    amount; unspent alpha is not carried forward.

    Args:
        looks: Strictly increasing positive cumulative event totals; the last
            is the final analysis.
        margin: Null efficacy margin (``H0: E <= margin``).
        alpha: One-sided type I error.
        rho: Treatment-to-control person-time ratio.
        spending: ``"obf"`` or ``"pocock"``.

    Returns:
        The design.

    Raises:
        ValueError: If ``looks`` is invalid.

    Examples:
        >>> design = design_group_sequential([40, 80], margin=0.3)
        >>> design.alpha_spent[-1] <= 0.025
        True
    """
    if not looks or looks[0] <= 0 or any(b <= a for a, b in pairwise(looks)):
        msg = "looks must be positive and strictly increasing"
        raise ValueError(msg)
    theta0 = efficacy_share(margin, rho)
    dist = np.array([1.0])
    previous, spent = 0, 0.0
    critical: list[int] = []
    spent_by_look: list[float] = []
    for look in looks:
        dist = _convolve(dist, look - previous, theta0)
        budget = alpha_spending(look / looks[-1], alpha, spending) - spent
        cumulative = np.cumsum(dist)
        allowed = np.flatnonzero(cumulative <= budget * (1 + 1e-12))
        c = int(allowed[-1]) if allowed.size else -1
        if c >= 0:
            spent += float(cumulative[c])
            dist[: c + 1] = 0.0
        critical.append(c)
        spent_by_look.append(spent)
        previous = look
    return GroupSequentialDesign(
        tuple(looks),
        tuple(critical),
        tuple(spent_by_look),
        margin,
        alpha,
        rho,
        spending,
    )


@dataclass(frozen=True)
class OperatingCharacteristics:
    """Exact stopping behaviour of a group-sequential design.

    Attributes:
        efficacy: Assumed true efficacy.
        efficacy_stop: Probability of stopping for efficacy at each look.
        futility_stop: Probability of stopping for futility at each look
            (always zero at the final look).
        power: Total probability of an efficacy conclusion.
        expected_events: Expected event total at stopping.
        futility_threshold: Interim futility rule (stop when the efficacy
            point estimate is at most this), or ``None``.
    """

    efficacy: float
    efficacy_stop: tuple[float, ...]
    futility_stop: tuple[float, ...]
    power: float
    expected_events: float
    futility_threshold: float | None


def operating_characteristics(
    design: GroupSequentialDesign,
    efficacy: float,
    *,
    futility_threshold: float | None = 0.0,
) -> OperatingCharacteristics:
    """Exact stopping probabilities, power and expected events.

    The futility rule is non-binding: boundaries ignore it, so type I error
    is controlled whether or not it is followed, and applying it here only
    lowers power.

    Args:
        design: Efficacy boundaries.
        efficacy: Assumed true efficacy.
        futility_threshold: Stop at an interim look when the point estimate
            is at most this; ``None`` disables futility stopping.

    Returns:
        Operating characteristics.

    Examples:
        >>> design = design_group_sequential([80], margin=0.3)
        >>> oc = operating_characteristics(design, 0.3, futility_threshold=None)
        >>> oc.power <= 0.025
        True
    """
    theta = efficacy_share(efficacy, design.rho)
    dist = np.array([1.0])
    previous = 0
    efficacy_stop: list[float] = []
    futility_stop: list[float] = []
    expected = 0.0
    final = len(design.looks) - 1
    for j, (look, c) in enumerate(zip(design.looks, design.critical, strict=True)):
        dist = _convolve(dist, look - previous, theta)
        stop = float(dist[: c + 1].sum()) if c >= 0 else 0.0
        if c >= 0:
            dist[: c + 1] = 0.0
        futile = 0.0
        if futility_threshold is not None and j < final:
            share = efficacy_share(futility_threshold, design.rho)
            # Estimate <= threshold  <=>  k / look >= share.
            first = math.ceil(share * look - 1e-12)
            futile = float(dist[first:].sum())
            dist[first:] = 0.0
        efficacy_stop.append(stop)
        futility_stop.append(futile)
        expected += look * (stop + futile)
        previous = look
    expected += design.looks[-1] * float(dist.sum())
    return OperatingCharacteristics(
        efficacy,
        tuple(efficacy_stop),
        tuple(futility_stop),
        float(sum(efficacy_stop)),
        expected,
        futility_threshold,
    )


def max_events_for_power(  # ruff: ignore[too-many-arguments]
    fractions: list[float],
    efficacy: float,
    *,
    power: float = 0.9,
    margin: float = 0.0,
    alpha: float = 0.025,
    rho: float = 1.0,
    spending: Literal["obf", "pocock"] = "obf",
    futility_threshold: float | None = 0.0,
    max_events: int = 2000,
) -> GroupSequentialDesign:
    """Smallest maximum event total whose group-sequential power meets ``power``.

    Looks are at ``round(fraction * total)``; the scan starts at the fixed
    design's event requirement minus 20%. Like fixed designs, discrete power
    can be non-monotone in the total, so the returned design is the first
    qualifying one.

    Args:
        fractions: Information fractions of each look, ending at 1.
        efficacy: Assumed true efficacy.
        power: Target power.
        margin: Null efficacy margin.
        alpha: One-sided type I error.
        rho: Treatment-to-control person-time ratio.
        spending: ``"obf"`` or ``"pocock"``.
        futility_threshold: Interim futility rule used for planning power.
        max_events: Search limit.

    Returns:
        The first qualifying design.

    Raises:
        ValueError: If no design up to ``max_events`` qualifies.

    Examples:
        >>> design = max_events_for_power([0.5, 1.0], 0.7, margin=0.3)
        >>> design.looks[-1] >= events_required(0.7, margin=0.3) - 10
        True
    """
    try:
        fixed = events_required(
            efficacy,
            power=power,
            rho=rho,
            margin=margin,
            confidence=1 - 2 * alpha,
            max_events=max_events,
        )
    except ValueError:
        fixed = 2
    for total in range(max(2, int(0.8 * fixed)), max_events + 1):
        looks = sorted({max(1, round(f * total)) for f in fractions})
        design = design_group_sequential(
            looks, margin=margin, alpha=alpha, rho=rho, spending=spending
        )
        oc = operating_characteristics(
            design, efficacy, futility_threshold=futility_threshold
        )
        if oc.power >= power:
            return design
    msg = f"No maximum event total up to {max_events} reaches power {power}"
    raise ValueError(msg)


@dataclass(frozen=True)
class AccrualPlan:
    """Enrolment, dropout and seasonality assumptions in calendar time.

    Attributes:
        enrolled: Total participants (both arms, 1:1).
        ramp_periods: Periods to complete uniform enrolment (0 = all at start).
        dropout_per_period: Loss-to-follow-up hazard per period.
        seasonal_amplitude: Relative amplitude of a cosine seasonal hazard,
            in [0, 1).
        peak_period: Period of peak hazard.
        season_length: Periods per seasonal cycle.
    """

    enrolled: int
    ramp_periods: float = 0.0
    dropout_per_period: float = 0.0
    seasonal_amplitude: float = 0.0
    peak_period: float = 0.0
    season_length: float = 12.0


def expected_events_by_period(
    plan: AccrualPlan,
    *,
    control_rate: float,
    efficacy: float,
    periods: int,
    substeps: int = 8,
) -> NDArray[np.float64]:
    """Expected cumulative events (both arms) at the end of each period.

    Args:
        plan: Enrolment and seasonality assumptions.
        control_rate: Control-arm event rate per participant per period.
        efficacy: True efficacy.
        periods: Horizon.
        substeps: Midpoint-rule substeps per period.

    Returns:
        Cumulative expected events at periods ``1..periods``.

    Examples:
        >>> plan = AccrualPlan(enrolled=1000)
        >>> events = expected_events_by_period(
        ...     plan, control_rate=0.01, efficacy=0.0, periods=2
        ... )
        >>> [round(float(x), 6) for x in events]
        [10.0, 20.0]
    """
    dt = 1.0 / substeps
    time = (np.arange(periods * substeps) + 0.5) * dt
    if plan.ramp_periods > 0:
        enrolled = plan.enrolled * np.clip(time / plan.ramp_periods, 0.0, 1.0)
    else:
        enrolled = np.full_like(time, float(plan.enrolled))
    survival = math.exp(-plan.dropout_per_period * dt)
    at_risk = np.empty_like(time)
    current, previous = 0.0, 0.0
    half = math.sqrt(survival)
    for i, n in enumerate(enrolled):
        current += n - previous
        previous = float(n)
        at_risk[i] = current * half
        current *= survival
    season = 1.0 + plan.seasonal_amplitude * np.cos(
        2 * math.pi * (time - plan.peak_period) / plan.season_length
    )
    arm_mean = 0.5 * (2.0 - efficacy)
    per_step = at_risk * control_rate * season * arm_mean * dt
    return np.asarray(np.cumsum(per_step)[substeps - 1 :: substeps], dtype=float)


def periods_to_events(cumulative: NDArray[np.float64], target: float) -> float:
    """First period at whose end expected cumulative events reach ``target``.

    Args:
        cumulative: Output of :func:`expected_events_by_period`.
        target: Event target.

    Returns:
        Period index (1-based), or ``inf`` if never reached.

    Examples:
        >>> periods_to_events(np.array([1.0, 2.0, 3.0]), 2.5)
        3.0
    """
    reached = np.flatnonzero(cumulative >= target)
    return float(reached[0] + 1) if reached.size else math.inf
