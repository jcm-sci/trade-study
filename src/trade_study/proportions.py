"""Fixed-sample binomial intervals and Wilson precision planning (#175)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil, isfinite, sqrt
from numbers import Integral, Real
from statistics import NormalDist
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .protocols import ResultsTable

_MAX_TRIALS = 2**53
_METHODS = ("wilson", "clopper-pearson")


@dataclass(frozen=True)
class ProportionInterval:
    """Two-sided, marginal interval for a fixed-sample Bernoulli probability.

    Wilson coverage is approximate; Clopper-Pearson is conservative under
    independent identically distributed Bernoulli sampling at a fixed n.
    Neither method supplies a sequential or post-selection guarantee.

    Attributes:
        k: Number of successes.
        n: Number of trials.
        estimate: Observed proportion k / n, not the Wilson interval center.
        low: Lower probability bound.
        high: Upper probability bound.
        confidence: Nominal two-sided confidence level.
        method: Wilson without continuity correction, or Clopper-Pearson.
    """

    k: int
    n: int
    estimate: float
    low: float
    high: float
    confidence: float
    method: str


def _count(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        msg = f"{name} must be a non-negative integer count, not a boolean"
        raise ValueError(msg)
    return int(value)


def _tail(confidence: float, method: str) -> float:
    if method not in _METHODS:
        msg = f"method must be one of {_METHODS}"
        raise ValueError(msg)
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, Real)
        or not isfinite(confidence)
        or not 0 < confidence < 1
    ):
        msg = "confidence must be finite and lie in (0, 1)"
        raise ValueError(msg)
    tail = (1 - float(confidence)) / 2
    if not 0 < tail < 0.5:
        msg = "confidence is too close to a floating-point boundary"
        raise ValueError(msg)
    return tail


def proportion_interval(
    k: int,
    n: int,
    *,
    confidence: float = 0.95,
    method: str = "wilson",
) -> ProportionInterval:
    """Estimate a binomial proportion and its two-sided fixed-sample interval.

    Trials must be independent Bernoulli outcomes with a common probability
    and a prespecified sample size. Callers establish these assumptions;
    counts alone cannot verify them. Intervals are marginal, without a
    multiplicity adjustment, and are not valid for arbitrary optional stopping.

    Args:
        k: Integer successes, including NumPy integer scalars; 0 <= k <= n.
        n: Positive integer trial count, at most 2**53 for float64 arithmetic.
        confidence: Nominal level in (0, 1), default 0.95.
        method: "wilson" without continuity correction (core), or
            "clopper-pearson" using the optional stats extra.

    Returns:
        Observed estimate and interval with counts and method recorded.

    Raises:
        ValueError: If counts, confidence or method are invalid or unsupported.
        ImportError: If Clopper-Pearson is requested without the stats extra.

    Examples:
        >>> interval = proportion_interval(8, 10)
        >>> interval.estimate, interval.n, interval.method
        (0.8, 10, 'wilson')
        >>> interval.low < interval.estimate < interval.high
        True
    """
    successes, trials = _count(k, "k"), _count(n, "n")
    if not 0 < trials <= _MAX_TRIALS or successes > trials:
        msg = "counts must satisfy 0 <= k <= n and 1 <= n <= 2**53"
        raise ValueError(msg)
    tail = _tail(confidence, method)
    estimate = successes / trials
    if method == "clopper-pearson":
        try:
            from scipy.special import (  # type: ignore[import-untyped]
                betainccinv,
                betaincinv,
            )
        except ImportError as error:
            msg = "Clopper-Pearson intervals require trade-study[stats] (SciPy >=1.11)"
            raise ImportError(msg) from error
        low = (
            0.0
            if successes == 0
            else float(betaincinv(successes, trials - successes + 1, tail))
        )
        high = (
            1.0
            if successes == trials
            else float(betainccinv(successes + 1, trials - successes, tail))
        )
        if not 0 <= low <= high <= 1:
            msg = "Clopper-Pearson beta inversion produced invalid bounds"
            raise ValueError(msg)
    else:
        z = -NormalDist().inv_cdf(tail)
        correction = z * z / trials
        center = (estimate + correction / 2) / (1 + correction)
        radius = (
            z
            * sqrt((estimate * (1 - estimate) + correction / 4) / trials)
            / (1 + correction)
        )
        low = 0.0 if successes == 0 else max(0.0, center - radius)
        high = 1.0 if successes == trials else min(1.0, center + radius)
    return ProportionInterval(
        successes, trials, estimate, low, high, float(confidence), method
    )


def replicates_for_half_width(half_width: float, *, confidence: float = 0.95) -> int:
    """Plan a fixed trial count from the worst-case Wilson interval half-width.

    The largest continuous Wilson half-width occurs at observed proportion
    1/2 and equals z / (2 * sqrt(n + z**2)). Inverting this expression gives
    the smallest positive integer satisfying that conservative bound, even
    when n is odd. This is interval-precision planning, not power analysis,
    a guarantee of Wilson coverage, or a rule for stopping after each trial.

    Args:
        half_width: Target (high - low) / 2 in (0, 0.5]. This is not the
            largest distance from the observed estimate to a bound.
        confidence: Nominal two-sided level in (0, 1).

    Returns:
        Planned integer n, at most 2**53. Fix n before collecting outcomes.

    Raises:
        ValueError: If inputs are invalid or the count exceeds float64 support.

    Examples:
        >>> replicates_for_half_width(0.05)
        381
        >>> replicates_for_half_width(0.10) < replicates_for_half_width(0.05)
        True
    """
    tail = _tail(confidence, "wilson")
    if (
        isinstance(half_width, bool)
        or not isinstance(half_width, Real)
        or not isfinite(half_width)
        or not 0 < half_width <= 0.5
    ):
        msg = "half_width must be finite and lie in (0, 0.5]"
        raise ValueError(msg)
    z = -NormalDist().inv_cdf(tail)
    if half_width < z / (2 * sqrt(_MAX_TRIALS + z * z)):
        msg = "Requested half_width needs more than 2**53 trials"
        raise ValueError(msg)
    count = min(_MAX_TRIALS, max(1, ceil((z / (2 * half_width)) ** 2 - z * z)))
    # Correct the integer rounding at an exactly representable target width.
    while count < _MAX_TRIALS and z / (2 * sqrt(count + z * z)) > half_width:
        count += 1
    while count > 1 and z / (2 * sqrt(count - 1 + z * z)) <= half_width:
        count -= 1
    return count


def _proportion_summaries(
    results: ResultsTable,
    groups: dict[int, list[int]],
    names: list[str],
    *,
    confidence: float,
    method: str,
) -> dict[int, dict[str, object]]:
    """Validate raw indicator rows and attach explicit marginal interval records.

    Returns:
        Interval records keyed by design point and observable.

    Raises:
        ValueError: If selected names, parallel rows, identities or scores are invalid.
    """
    _tail(confidence, method)
    if not results.configs:
        msg = "Indicator summaries require nonempty trial rows"
        raise ValueError(msg)
    if not names or len(names) != len(set(names)):
        msg = "proportion_observables must be nonempty and unique"
        raise ValueError(msg)
    if len(results.observable_names) != len(set(results.observable_names)):
        msg = "Indicator summaries require unique observable names"
        raise ValueError(msg)
    if results.scores.shape != (
        len(results.configs),
        len(results.observable_names),
    ) or len(results.metadata) != len(results.configs):
        msg = "Indicator summaries require matching score/config/metadata rows"
        raise ValueError(msg)
    unknown = set(names) - set(results.observable_names)
    if unknown:
        msg = f"Unknown proportion_observables: {sorted(unknown)}"
        raise ValueError(msg)
    summaries: dict[int, dict[str, object]] = {}
    for design, rows in groups.items():
        reps = [_count(results.metadata[row].get("rep"), "rep") for row in rows]
        if len(reps) != len(set(reps)):
            msg = "Indicator summaries require unique replicate identities per design"
            raise ValueError(msg)
        intervals: dict[str, object] = {}
        for name in names:
            values = results.scores[rows, results.observable_names.index(name)]
            if not np.all((values == 0) | (values == 1)):
                msg = (
                    f"Indicator {name!r} requires finite 0/1 outcomes; "
                    "no rows are dropped"
                )
                raise ValueError(msg)
            interval = proportion_interval(
                int(np.count_nonzero(values)),
                len(rows),
                confidence=confidence,
                method=method,
            )
            intervals[name] = {**asdict(interval), "scope": "fixed-sample-marginal"}
        summaries[design] = intervals
    return summaries
