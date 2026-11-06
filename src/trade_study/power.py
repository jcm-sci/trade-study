"""Monte Carlo power curves and minimum adequate design sizes.

A power curve orders the per-design success probability of a raw 0/1
indicator observable (for example, "the simulated trial rejected the null")
along one size factor, separately for every combination of the other
factors. ``minimum_adequate`` then reports the smallest size whose power
meets a target under an explicit rule.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import pairwise
from operator import itemgetter
from typing import TYPE_CHECKING, Any, Literal

from .proportions import ProportionInterval, proportion_interval

if TYPE_CHECKING:
    from .protocols import ResultsTable

_RULES = ("lower_bound", "estimate")
_ADJUSTMENTS = (None, "bonferroni")


@dataclass(frozen=True)
class PowerCurve:
    """Success-probability intervals for one stratum along a size factor.

    Attributes:
        indicator: Name of the 0/1 indicator observable.
        along: Name of the ordered size factor.
        stratum: Values of every other configuration factor.
        values: Sorted, distinct values of ``along`` in this stratum.
        intervals: Fixed-sample interval at each value, in ``values`` order.
        confidence: Two-sided level used for each interval.
        family_confidence: Nominal simultaneous level across this stratum's
            values; equals ``confidence`` when ``adjustment`` is ``None``.
        adjustment: ``None`` (marginal intervals) or ``"bonferroni"``.
    """

    indicator: str
    along: str
    stratum: dict[str, Any]
    values: tuple[Any, ...]
    intervals: tuple[ProportionInterval, ...]
    confidence: float
    family_confidence: float
    adjustment: str | None


@dataclass(frozen=True)
class AdequateDesign:
    """Smallest size along a power curve that meets a target.

    Attributes:
        stratum: Values of every other configuration factor.
        along: Name of the ordered size factor.
        value: Selected size, or ``None`` when no scanned size qualifies.
        interval: Interval at the selected size, or ``None``.
        target: Required success probability.
        rule: ``"lower_bound"`` or ``"estimate"``.
        monotone_above: Whether every larger scanned size also qualifies.
            Always ``False`` when ``value`` is ``None``.
    """

    stratum: dict[str, Any]
    along: str
    value: Any
    interval: ProportionInterval | None
    target: float
    rule: str
    monotone_above: bool


def _stratum_key(stratum: dict[str, Any]) -> str:
    return json.dumps(stratum, sort_keys=True, default=repr)


def power_curve(
    results: ResultsTable,
    indicator: str,
    along: str,
    *,
    confidence: float = 0.95,
    method: str = "wilson",
    adjustment: Literal["bonferroni"] | None = None,
) -> list[PowerCurve]:
    """Build per-stratum power curves from raw replicate rows.

    Rows are grouped by design point exactly as in
    :meth:`~trade_study.protocols.ResultsTable.aggregate_replicates`, which
    validates the 0/1 indicator and unique replicate identities. Each
    interval is a fixed-sample interval for i.i.d. replicates. With
    ``adjustment="bonferroni"`` each stratum's intervals use level
    ``1 - (1 - confidence) / m`` for its ``m`` scanned values, so that
    selecting a size from the curve keeps nominal ``confidence``
    simultaneously across those values.

    Args:
        results: Raw (unaggregated) replicate rows from ``run_grid``.
        indicator: Name of a 0/1 indicator observable.
        along: Configuration factor that orders the curve (for example a
            sample size). Values must be mutually orderable.
        confidence: Two-sided family level in (0, 1).
        method: ``"wilson"`` or ``"clopper-pearson"``.
        adjustment: ``None`` for marginal intervals or ``"bonferroni"``.

    Returns:
        One curve per combination of the remaining factors, in first-seen
        order.

    Raises:
        ValueError: If ``along`` is missing, values cannot be ordered, a
            stratum repeats a value, or ``adjustment`` is unsupported.

    Examples:
        >>> import numpy as np
        >>> from trade_study import ResultsTable
        >>> rows = [(n, r) for n in (10, 20) for r in range(4)]
        >>> table = ResultsTable(
        ...     [{"n": n} for n, _ in rows],
        ...     np.array([[float(n == 20 or r < 2)] for n, r in rows]),
        ...     ["reject"],
        ...     metadata=[{"design_point": n, "rep": r} for n, r in rows],
        ... )
        >>> [c.estimate for c in power_curve(table, "reject", "n")[0].intervals]
        [0.5, 1.0]
    """
    if adjustment not in _ADJUSTMENTS:
        msg = f"adjustment must be one of {_ADJUSTMENTS}"
        raise ValueError(msg)
    summary = results.aggregate_replicates(
        proportion_observables=[indicator],
        confidence=confidence,
        proportion_method=method,
    )
    strata: dict[str, tuple[dict[str, Any], list[tuple[Any, int, int]]]] = {}
    for config, meta in zip(summary.configs, summary.metadata, strict=True):
        if along not in config:
            msg = f"Factor {along!r} is absent from configuration {config!r}"
            raise ValueError(msg)
        stratum = {k: v for k, v in config.items() if k != along}
        record = meta["proportion_intervals"][indicator]
        entry = strata.setdefault(_stratum_key(stratum), (stratum, []))
        entry[1].append((config[along], int(record["k"]), int(record["n"])))

    curves: list[PowerCurve] = []
    for stratum, points in strata.values():
        try:
            points.sort(key=itemgetter(0))
        except TypeError as error:
            msg = f"Values of {along!r} must be mutually orderable"
            raise ValueError(msg) from error
        values = tuple(point[0] for point in points)
        if any(a == b for a, b in pairwise(values)):
            msg = f"Stratum {stratum!r} repeats a value of {along!r}"
            raise ValueError(msg)
        level = (
            1 - (1 - confidence) / len(points)
            if adjustment == "bonferroni"
            else confidence
        )
        intervals = tuple(
            proportion_interval(k, n, confidence=level, method=method)
            for _, k, n in points
        )
        curves.append(
            PowerCurve(
                indicator=indicator,
                along=along,
                stratum=stratum,
                values=values,
                intervals=intervals,
                confidence=float(level),
                family_confidence=float(confidence),
                adjustment=adjustment,
            )
        )
    return curves


def minimum_adequate(
    curves: list[PowerCurve],
    target: float,
    *,
    rule: Literal["lower_bound", "estimate"] = "lower_bound",
) -> list[AdequateDesign]:
    """Select the smallest size on each curve whose power meets ``target``.

    ``"lower_bound"`` requires the interval's lower bound to reach the
    target; ``"estimate"`` uses the observed proportion. A curve that
    dips below the target above the selection is reported through
    ``monotone_above=False`` rather than smoothed. Unless the curves were
    built with ``adjustment="bonferroni"``, the lower bound is marginal at
    the selected size only.

    Args:
        curves: Output of :func:`power_curve`.
        target: Required success probability in [0, 1].
        rule: ``"lower_bound"`` (default) or ``"estimate"``.

    Returns:
        One selection per curve, in the same order.

    Raises:
        ValueError: If ``target`` or ``rule`` is invalid.

    Examples:
        >>> import numpy as np
        >>> from trade_study import ResultsTable
        >>> rows = [(n, r) for n in (10, 20) for r in range(4)]
        >>> table = ResultsTable(
        ...     [{"n": n} for n, _ in rows],
        ...     np.array([[float(n == 20 or r < 2)] for n, r in rows]),
        ...     ["reject"],
        ...     metadata=[{"design_point": n, "rep": r} for n, r in rows],
        ... )
        >>> curves = power_curve(table, "reject", "n")
        >>> minimum_adequate(curves, 0.8, rule="estimate")[0].value
        20
    """
    if rule not in _RULES:
        msg = f"rule must be one of {_RULES}"
        raise ValueError(msg)
    if not 0 <= target <= 1:
        msg = "target must lie in [0, 1]"
        raise ValueError(msg)

    def meets(interval: ProportionInterval) -> bool:
        statistic = interval.low if rule == "lower_bound" else interval.estimate
        return statistic >= target

    selections: list[AdequateDesign] = []
    for curve in curves:
        passing = [meets(interval) for interval in curve.intervals]
        index = next((i for i, ok in enumerate(passing) if ok), None)
        selections.append(
            AdequateDesign(
                stratum=curve.stratum,
                along=curve.along,
                value=None if index is None else curve.values[index],
                interval=None if index is None else curve.intervals[index],
                target=float(target),
                rule=rule,
                monotone_above=index is not None and all(passing[index:]),
            )
        )
    return selections
