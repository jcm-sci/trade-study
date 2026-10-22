"""One explicit held-out partition for shared datasets and conditioned surfaces."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from trade_study import (
    Factor,
    FactorType,
    ResultsTable,
    fit_surrogate,
    sensitivity_from_table,
)


@pytest.mark.parametrize("method", ["rf", "gp"])
def test_external_groups_reuse_held_out_cv_and_merge_equal_numeric_labels(
    method: str,
) -> None:
    table = ResultsTable(
        [{"x": float(i)} for i in range(6) for _ in range(6)],
        np.repeat([-3, 8, -5, 1, 9, -8], 6).reshape(-1, 1),
        ["loss"],
    )
    labels = [
        ("dataset", float(i) if rep % 2 else np.int64(i))
        for i in range(6)
        for rep in range(6)
    ]
    fitted = fit_surrogate(
        table,
        [Factor("x", FactorType.CONTINUOUS, bounds=(0, 5))],
        method=method,
        cv_groups=labels,
        cv_folds=3,
        n_estimators=30,
        warn_below_r2=None,
    )
    assert fitted.cv_group_source == "external"
    assert fitted.cv_group_by is None
    assert fitted.cv_group_labels == tuple(labels)
    assert fitted.validation_groups == [i for i in range(6) for _ in range(6)]
    assert fitted.row_cv_r2["loss"] > 0.95
    assert fitted.cv_r2["loss"] < fitted.row_cv_r2["loss"] - 0.5


def _shared_table() -> tuple[ResultsTable, list[Factor], list[object]]:
    """Create a shared-dataset partition crossing all configurations.

    Returns:
        q-by-x raw replicates, factors and original-row dataset identities.
    """
    configs = [
        {"q": q, "x": x} for q in (2, 4) for x in (0.0, 0.5, 1.0) for _ in range(4)
    ]
    labels: list[object] = [rep for _q in (2, 4) for _x in range(3) for rep in range(4)]
    return (
        ResultsTable(
            configs,
            np.array([
                [config["q"] * config["x"] + 0.01 * int(rep)]
                for config, rep in zip(configs, labels, strict=True)
            ]),
            ["loss"],
            metadata=[{"dataset": label} for label in labels],
        ),
        [
            Factor("q", FactorType.DISCRETE, levels=[2, 4]),
            Factor("x", FactorType.CONTINUOUS, bounds=(0, 1)),
        ],
        labels,
    )


@pytest.mark.parametrize("vector", ["python", "numpy"])
def test_conditioned_shared_groups_preserve_original_alignment_and_partition(
    vector: str,
) -> None:
    table, factors, labels = _shared_table()
    fitted = sensitivity_from_table(
        table,
        factors,
        condition_on={"q": 4},
        cv_groups=labels if vector == "python" else np.asarray(labels),
        cv_folds=2,
        n_estimators=20,
        n_trajectories=32,
        warn_below_r2=None,
    )
    assert fitted.source_rows == tuple(range(12, 24))
    assert fitted.cv_group_source == "external"
    assert fitted.validation_scheme == "held-out-external-groups"
    assert fitted.cv_group_by is None
    assert fitted.cv_group_labels == tuple(labels[12:])
    assert fitted.validation_groups == [0, 1, 2, 3] * 3
    assert fitted.importance["loss"][0] == pytest.approx(1, abs=0.12)
    assert np.isfinite(fitted.surrogate_cv_rmse["loss"])


def test_one_surviving_external_group_never_falls_back_to_row_metrics() -> None:
    table, factors, labels = _shared_table()
    labels[12:] = ["one-dataset"] * 12
    fitted = sensitivity_from_table(
        table,
        factors,
        condition_on={"q": 4},
        cv_groups=labels,
        n_estimators=20,
        n_trajectories=16,
        warn_below_r2=None,
    )
    assert fitted.validation_groups == [0] * 12
    assert fitted.surrogate_row_cv_r2["loss"] > 0.9
    assert np.isnan(fitted.surrogate_cv_r2["loss"])
    assert np.isnan(fitted.surrogate_cv_rmse["loss"])


@pytest.mark.parametrize(
    "label", [None, float("nan"), float("inf"), [1], ("dataset", None), complex(1, 2)]
)
def test_invalid_labels_are_refused_even_in_rows_excluded_by_conditioning(
    label: object,
) -> None:
    table, factors, labels = _shared_table()
    labels[0] = label
    with pytest.raises(ValueError, match="group labels"):
        sensitivity_from_table(table, factors, condition_on={"q": 4}, cv_groups=labels)


@pytest.mark.parametrize("labels", [[], [1], "one-label", [1] * 25])
def test_original_group_vector_length_is_checked(labels: Any) -> None:
    table, factors, _ = _shared_table()
    with pytest.raises(ValueError, match="per original row"):
        sensitivity_from_table(table, factors, condition_on={"q": 4}, cv_groups=labels)


def test_group_sources_are_mutually_exclusive() -> None:
    table, factors, labels = _shared_table()
    with pytest.raises(ValueError, match="mutually exclusive"):
        sensitivity_from_table(
            table,
            factors,
            condition_on={"q": 4},
            cv_groups=labels,
            cv_group_by="design",
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        fit_surrogate(table, factors, cv_groups=labels, cv_group_by="design")


@pytest.mark.parametrize(
    "labels",
    [
        dict.fromkeys(range(24), "shared-dataset"),
        set(range(24)),
        frozenset(range(24)),
        np.ones((24, 1)),
        np.asarray(1),
    ],
)
def test_group_vectors_refuse_mapping_unordered_and_nonvector_inputs(
    labels: Any,
) -> None:
    table, factors, _ = _shared_table()
    with pytest.raises(ValueError, match="ordered vector"):
        fit_surrogate(table, factors, cv_groups=labels)
    with pytest.raises(ValueError, match="ordered vector"):
        sensitivity_from_table(table, factors, condition_on={"q": 4}, cv_groups=labels)
