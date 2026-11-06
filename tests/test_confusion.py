"""Tests for confusion tables of replicated outcomes (#235)."""

from __future__ import annotations

import pytest

from trade_study import ConfusionTable, confusion_table, proportion_interval

ROWS = ["A", "B"]
COLUMNS = ["A", "B", "none", "failed"]


def _table(**kwargs: object) -> ConfusionTable:
    truth = ["A"] * 6 + ["B"] * 4
    verdict = ["A", "A", "A", "none", "none", "failed", "B", "B", "A", "none"]
    return confusion_table(truth, verdict, ROWS, COLUMNS, **kwargs)  # type: ignore[arg-type]


def test_counts_totals_and_declared_order() -> None:
    table = _table()
    assert table.rows == ("A", "B")
    assert table.columns == tuple(COLUMNS)
    assert table.counts == ((3, 0, 2, 1), (1, 2, 1, 0))
    assert table.row_total("A") == 6
    assert table.row_total("B") == 4


def test_cells_equal_proportion_interval_for_the_same_counts() -> None:
    table = _table()
    for i, row in enumerate(table.rows):
        for j, column in enumerate(table.columns):
            expected = proportion_interval(table.counts[i][j], table.row_total(row))
            assert table.cell(row, column) == expected
            assert table.intervals[i][j].n == table.row_total(row)


def test_row_proportions_sum_to_one() -> None:
    table = _table()
    for row in table.rows:
        total = sum(table.cell(row, column).estimate for column in table.columns)
        assert total == pytest.approx(1.0)


def test_bonferroni_level_and_wider_intervals() -> None:
    marginal, family = _table(), _table(adjustment="bonferroni")
    assert marginal.confidence == pytest.approx(0.95)
    assert marginal.adjustment is None
    assert family.confidence == pytest.approx(1 - 0.05 / 4)
    assert family.family_confidence == pytest.approx(0.95)
    assert family.adjustment == "bonferroni"
    a, b = marginal.cell("A", "A"), family.cell("A", "A")
    assert b.high - b.low > a.high - a.low


def test_clopper_pearson_method_is_forwarded() -> None:
    pytest.importorskip("scipy")
    table = _table(method="clopper-pearson")
    assert table.method == "clopper-pearson"
    assert table.cell("A", "A").method == "clopper-pearson"


def test_a_failed_trial_is_a_column_not_a_dropped_cohort() -> None:
    table = _table()
    assert sum(sum(row) for row in table.counts) == 10


@pytest.mark.parametrize(
    ("truth", "verdict", "rows", "columns", "pattern"),
    [
        (["A"], ["A", "B"], ROWS, COLUMNS, "same length"),
        (["A", "C"], ["A", "A"], ROWS, COLUMNS, "not declared"),
        (["A", "B"], ["A", "other"], ROWS, COLUMNS, "not declared"),
        (["A", "A"], ["A", "B"], ROWS, COLUMNS, "without a cohort"),
        (["A", "B"], ["A", "B"], ["A", "A"], COLUMNS, "repeated"),
        (["A", "B"], ["A", "B"], ROWS, [], "must not be empty"),
    ],
)
def test_invalid_input_is_refused(
    truth: list[str],
    verdict: list[str],
    rows: list[str],
    columns: list[str],
    pattern: str,
) -> None:
    with pytest.raises(ValueError, match=pattern):
        confusion_table(truth, verdict, rows, columns)


def test_unsupported_adjustment_is_refused() -> None:
    with pytest.raises(ValueError, match="adjustment"):
        _table(adjustment="holm")


def test_bonferroni_cells_equal_proportion_interval_at_the_adjusted_level() -> None:
    table = _table(adjustment="bonferroni")
    level = 1 - 0.05 / len(COLUMNS)
    for i, row in enumerate(table.rows):
        for j, column in enumerate(table.columns):
            expected = proportion_interval(
                table.counts[i][j], table.row_total(row), confidence=level
            )
            assert table.cell(row, column) == expected


def test_a_single_column_has_the_marginal_level() -> None:
    table = confusion_table(
        ["A", "A"], ["A", "A"], ["A"], ["A"], adjustment="bonferroni"
    )
    assert table.confidence == pytest.approx(0.95)


def test_equal_comparing_labels_are_the_same_label() -> None:
    table = confusion_table([True, False], [1, 0], rows=[1, 0], columns=[True, False])
    assert table.counts == ((1, 0), (0, 1))


@pytest.mark.parametrize(
    ("truth", "verdict", "rows", "columns", "pattern"),
    [
        ([[1], [2]], ["A", "B"], ROWS, COLUMNS, "hashable"),
        (["A", "B"], ["A", "B"], [[1], [2]], COLUMNS, "hashable"),
        (["A"], ["A"], "A", COLUMNS, "not a single string"),
        (["A"], ["A"], ROWS, "AB", "not a single string"),
        ([], [], ROWS, COLUMNS, "without a cohort"),
    ],
)
def test_malformed_labels_raise_value_error(
    truth: list[object],
    verdict: list[object],
    rows: object,
    columns: object,
    pattern: str,
) -> None:
    with pytest.raises(ValueError, match=pattern):
        confusion_table(truth, verdict, rows, columns)  # type: ignore[arg-type]


def test_unknown_cell_labels_name_the_label_and_the_table_is_frozen() -> None:
    table = _table()
    with pytest.raises(ValueError, match="'C' is not a declared row"):
        table.cell("C", "A")
    with pytest.raises(ValueError, match="'zzz' is not a declared column"):
        table.cell("A", "zzz")
    with pytest.raises(ValueError, match="not a declared row"):
        table.row_total("C")
    with pytest.raises(AttributeError):
        table.rows = ()  # type: ignore[misc]


def test_invalid_confidence_and_method_are_refused() -> None:
    with pytest.raises(ValueError, match="confidence"):
        _table(confidence=1.0)
    with pytest.raises(ValueError, match="method"):
        _table(method="exact")
