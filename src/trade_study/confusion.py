"""Confusion tables for replicated recovery and discrimination studies.

A study that generates independent cohorts from each candidate structure and
reads every cohort with a prespecified rule (for example "favour A", "favour
B", "indistinguishable", "failed") produces a table of counts with one row
per generating structure and one column per verdict. ``confusion_table``
reports that table with row proportions and a fixed-sample interval for every
cell, so the wrong-verdict and no-verdict rates are visible beside the
probability of the right verdict.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from .proportions import ProportionInterval, proportion_interval

if TYPE_CHECKING:
    from collections.abc import Hashable, Sequence

_ADJUSTMENTS = (None, "bonferroni")


@dataclass(frozen=True)
class ConfusionTable:
    """Counts and row-wise proportion intervals of verdicts by truth.

    Attributes:
        rows: Row labels (the truth), in the order declared by the caller.
        columns: Column labels (the verdicts), in declared order.
        counts: ``counts[i][j]`` is the number of cohorts of row ``i`` with
            verdict ``j``.
        intervals: ``intervals[i][j]`` is the fixed-sample interval for the
            probability of verdict ``j`` given row ``i``; its ``n`` is the
            row total.
        confidence: Two-sided level of each reported interval (after any
            adjustment).
        family_confidence: Nominal simultaneous level across the columns of a
            row; equals ``confidence`` when ``adjustment`` is ``None``.
        adjustment: ``None`` (marginal intervals) or ``"bonferroni"``.
        method: Interval method passed to ``proportion_interval``.
    """

    rows: tuple[Hashable, ...]
    columns: tuple[Hashable, ...]
    counts: tuple[tuple[int, ...], ...]
    intervals: tuple[tuple[ProportionInterval, ...], ...]
    confidence: float
    family_confidence: float
    adjustment: str | None
    method: str

    def cell(self, row: Hashable, column: Hashable) -> ProportionInterval:
        """Interval for the probability of ``column`` given ``row``.

        Returns:
            The fixed-sample interval of that cell.
        """
        return self.intervals[self.rows.index(row)][self.columns.index(column)]

    def row_total(self, row: Hashable) -> int:
        """Number of cohorts of ``row``, including failed trials.

        Returns:
            The row total.
        """
        return sum(self.counts[self.rows.index(row)])


def _unique(labels: Sequence[Hashable], name: str) -> tuple[Hashable, ...]:
    declared = tuple(labels)
    if not declared:
        msg = f"{name} must not be empty"
        raise ValueError(msg)
    if len(set(declared)) != len(declared):
        msg = f"{name} contains a repeated label"
        raise ValueError(msg)
    return declared


def confusion_table(
    truth: Sequence[Hashable],
    verdict: Sequence[Hashable],
    rows: Sequence[Hashable],
    columns: Sequence[Hashable],
    *,
    confidence: float = 0.95,
    method: str = "wilson",
    adjustment: Literal["bonferroni"] | None = None,
) -> ConfusionTable:
    """Tabulate one verdict per independent cohort against its truth.

    Each position of ``truth`` and ``verdict`` is one independent cohort.
    Every label must be declared in ``rows`` or ``columns`` and every declared
    row must have at least one cohort, so no cohort is dropped and a failed
    trial is a column of its own. Each cell interval is a fixed-sample
    interval for independent cohorts of a prespecified number per row; the
    caller establishes the independence and the number (see
    ``proportion_interval``). With ``adjustment="bonferroni"`` the intervals of
    a row use level ``1 - (1 - confidence) / len(columns)`` so that the
    statements about one row hold simultaneously at the nominal level.

    Args:
        truth: Generating label of each cohort.
        verdict: Verdict label of each cohort, aligned with ``truth``.
        rows: Declared truth labels.
        columns: Declared verdict labels, including a failure label if the
            study can fail.
        confidence: Two-sided family level in (0, 1).
        method: ``"wilson"`` or ``"clopper-pearson"``.
        adjustment: ``None`` for marginal intervals or ``"bonferroni"``.

    Returns:
        The counts and the interval of every cell.

    Raises:
        ValueError: If lengths differ, labels are repeated or undeclared, a
            declared row has no cohort, or ``adjustment`` is unsupported.

    Examples:
        >>> table = confusion_table(
        ...     ["A", "A", "A", "B"],
        ...     ["A", "A", "none", "B"],
        ...     rows=["A", "B"],
        ...     columns=["A", "B", "none"],
        ... )
        >>> table.counts
        ((2, 0, 1), (0, 1, 0))
        >>> table.cell("A", "A").estimate
        0.6666666666666666
    """
    if adjustment not in _ADJUSTMENTS:
        msg = f"adjustment must be one of {_ADJUSTMENTS}"
        raise ValueError(msg)
    if len(truth) != len(verdict):
        msg = "truth and verdict must have the same length"
        raise ValueError(msg)
    row_labels = _unique(rows, "rows")
    column_labels = _unique(columns, "columns")
    undeclared = (set(truth) - set(row_labels)) | (set(verdict) - set(column_labels))
    if undeclared:
        msg = f"labels not declared in rows or columns: {sorted(map(repr, undeclared))}"
        raise ValueError(msg)
    tally = [[0] * len(column_labels) for _ in row_labels]
    for t, v in zip(truth, verdict, strict=True):
        tally[row_labels.index(t)][column_labels.index(v)] += 1
    empty = [
        row for row, counts in zip(row_labels, tally, strict=True) if not sum(counts)
    ]
    if empty:
        msg = f"declared rows without a cohort: {sorted(map(repr, empty))}"
        raise ValueError(msg)
    level = (
        1 - (1 - confidence) / len(column_labels)
        if adjustment == "bonferroni"
        else confidence
    )
    intervals = tuple(
        tuple(
            proportion_interval(k, sum(counts), confidence=level, method=method)
            for k in counts
        )
        for counts in tally
    )
    return ConfusionTable(
        rows=row_labels,
        columns=column_labels,
        counts=tuple(tuple(counts) for counts in tally),
        intervals=intervals,
        confidence=float(level),
        family_confidence=float(confidence),
        adjustment=adjustment,
        method=method,
    )
