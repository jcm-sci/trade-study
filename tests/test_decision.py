"""Explicit preferences, normalization, raw equivalence, and paired summaries."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from trade_study import (
    Constraint,
    Direction,
    Observable,
    PreferencePolicy,
    ResultsTable,
    preference_sweep,
)

_OBS = [
    Observable("cost", Direction.MINIMIZE),
    Observable("accuracy", Direction.MAXIMIZE),
]


def _table() -> ResultsTable:
    return ResultsTable(
        [{"design": "a"}, {"design": "b"}, {"design": "c"}],
        np.array([[100, 0.9], [200, 0.99], [300, 0.8]]),
        ["cost", "accuracy"],
    )


def test_mixed_units_directions_and_preference_sensitive_choices() -> None:
    policy = PreferencePolicy(
        [{"accuracy": 1}, {"cost": 1}, {"cost": 1, "accuracy": 1}],
        "minmax",
    )
    result = preference_sweep(_table(), _OBS, policy=policy)
    np.testing.assert_array_equal(result.pareto, [True, True, False])
    np.testing.assert_array_equal(result.ranks, [[2, 1, 3], [1, 2, 3], [1, 2, 3]])
    selection = [m["decision"]["selection_fraction"] for m in result.summary.metadata]
    np.testing.assert_allclose(selection, [2 / 3, 1 / 3, 0])
    assert result.normalization_bounds == {"cost": (100, 300), "accuracy": (0.8, 0.99)}
    scaled = _table()
    scaled.scores[:, 0] *= 1000
    scaled.scores[:, 1] *= 100
    other_units = preference_sweep(scaled, _OBS, policy=policy)
    np.testing.assert_array_equal(result.ranks, other_units.ranks)
    frame = result.summary.to_dataframe()
    assert "meta.decision.selection_fraction" in frame.columns
    assert result.summary.metadata[0]["decision"]["best_rank"] == 1
    assert result.summary.metadata[0]["decision"]["worst_rank"] == 2


def test_normalization_is_explicit_and_reference_values_are_not_clipped() -> None:
    weights = [{"cost": 0.05, "accuracy": 0.95}]
    raw = preference_sweep(_table(), _OBS, policy=PreferencePolicy(weights, "none"))
    normalized = preference_sweep(
        _table(), _OBS, policy=PreferencePolicy(weights, "minmax")
    )
    assert np.argmin(raw.utilities[0]) == 0
    assert np.argmin(normalized.utilities[0]) == 1
    assert raw.normalization_bounds == {}
    reference = preference_sweep(
        _table(),
        _OBS,
        policy=PreferencePolicy(
            [{"cost": 1}],
            "reference",
            reference_bounds={"cost": (0, 100), "accuracy": (0, 1)},
        ),
    )
    np.testing.assert_array_equal(reference.utilities[0], [1, 2, 3])


def test_weight_perturbations_ties_and_shared_selection_credit() -> None:
    table = ResultsTable(
        [{"design": "a"}, {"design": "b"}],
        np.array([[0.0, 0.0], [1.0, 1.0]]),
        ["cost", "accuracy"],
    )
    policy = PreferencePolicy(
        [
            {"cost": 0.49, "accuracy": 0.51},
            {"cost": 0.5, "accuracy": 0.5},
            {"cost": 0.51, "accuracy": 0.49},
        ],
        "minmax",
    )
    result = preference_sweep(table, _OBS, policy=policy)
    np.testing.assert_array_equal(result.ranks, [[2, 1], [1, 1], [1, 2]])
    assert [m["decision"]["selection_fraction"] for m in result.summary.metadata] == [
        0.5,
        0.5,
    ]
    # Policy is copied so later caller mutation cannot change reported assumptions.
    original = result.policy.weights[0]["cost"]
    policy.weights[0]["cost"] = 999
    table.scores[0, 0] = 10
    assert result.summary.scores[0, 0] == 0
    assert result.policy.weights[0]["cost"] == original


def test_zero_range_objectives_and_pairwise_raw_unit_equivalence() -> None:
    table = ResultsTable(
        [{"design": i} for i in range(3)],
        np.array([[10, 0.9], [10, 0.905], [10, 0.8]]),
        ["cost", "accuracy"],
    )
    result = preference_sweep(
        table,
        _OBS,
        policy=PreferencePolicy(
            [{"cost": 1}], "minmax", equivalence={"accuracy": 0.01}
        ),
    )
    np.testing.assert_array_equal(result.ranks, np.ones((1, 3)))
    assert result.equivalent[0, 1]
    assert not result.equivalent[0, 2]
    assert result.summary.metadata[0]["decision"][
        "selection_fraction"
    ] == pytest.approx(1 / 3)
    chain = ResultsTable(
        [{"x": x} for x in (0, 1, 2)], np.array([[0], [0.09], [0.18]]), ["cost"]
    )
    pairwise = preference_sweep(
        chain,
        _OBS[:1],
        policy=PreferencePolicy([{"cost": 1}], "none", equivalence={"cost": 0.1}),
    )
    assert pairwise.equivalent[0, 1]
    assert pairwise.equivalent[1, 2]
    assert not pairwise.equivalent[0, 2]


def test_infeasible_and_nonfinite_designs_do_not_set_normalization_anchors() -> None:
    table = _table()
    table.scores[2, 1] = np.nan
    result = preference_sweep(
        table,
        _OBS,
        policy=PreferencePolicy([{"cost": 1}], "minmax"),
        constraints=[Constraint("minimum_cost", "cost", ">=", 150)],
    )
    np.testing.assert_array_equal(result.feasible, [False, True, False])
    assert result.normalization_bounds["cost"] == (200, 200)
    assert np.isnan(result.ranks[0, 0])
    assert np.isnan(result.ranks[0, 2])
    assert result.summary.metadata[1]["decision"]["selection_fraction"] == 1
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        none = preference_sweep(
            table,
            _OBS,
            policy=PreferencePolicy([{"cost": 1}], "minmax"),
            constraints=[Constraint("impossible", "cost", "<", 0)],
        )
    assert not np.any(none.feasible)
    assert np.all(np.isnan(none.ranks))
    assert all(m["decision"]["selection_fraction"] == 0 for m in none.summary.metadata)


def test_adaptive_raw_means_override_weighted_columns() -> None:
    table = ResultsTable(
        [{"x": 0}, {"x": 1}],
        np.array([[1000.0, 1.0], [2000.0, 2.0]]),
        ["cost", "accuracy"],
        metadata=[
            {"scores": {"cost": 1, "accuracy": 1}},
            {"scores": {"cost": 2, "accuracy": 2}},
        ],
    )
    result = preference_sweep(
        table,
        [Observable("cost", Direction.MINIMIZE, weight=1000), _OBS[1]],
        policy=PreferencePolicy([{"cost": 1}], "none"),
        constraints=[Constraint("raw_cost", "cost", "<=", 1.5)],
    )
    np.testing.assert_array_equal(result.summary.scores, [[1, 1], [2, 2]])
    np.testing.assert_array_equal(result.feasible, [True, False])


def test_replicated_means_and_existing_paired_uncertainty() -> None:
    rows = [(design, rep) for design in range(2) for rep in range(4)]
    table = ResultsTable(
        [{"design": d} for d, _ in rows],
        np.array([[100 + 10 * d + rep, 0.8 + 0.1 * d + rep * 0.01] for d, rep in rows]),
        ["cost", "accuracy"],
        metadata=[{"design_point": d, "rep": rep} for d, rep in rows],
    )
    result = preference_sweep(
        table,
        _OBS,
        policy=PreferencePolicy(
            [{"accuracy": 1}], "minmax", paired_confidence=0.9, n_boot=50
        ),
        paired_reference=0,
    )
    assert len(result.summary.configs) == 2
    assert result.summary.metadata[0]["n_reps"] == 4
    assert result.paired["cost"][0].mean == pytest.approx(10)
    assert result.paired["accuracy"][0].mean == pytest.approx(0.1)
    assert result.paired["accuracy"][0].confidence == pytest.approx(0.95)
    with pytest.raises(ValueError, match="raw design_point/rep"):
        preference_sweep(
            _table(),
            _OBS,
            policy=PreferencePolicy([{"cost": 1}], "none"),
            paired_reference=0,
        )


@pytest.mark.parametrize(
    "policy",
    [
        PreferencePolicy([], "minmax"),
        PreferencePolicy([{"other": 1}], "minmax"),
        PreferencePolicy([{"cost": -1}], "minmax"),
        PreferencePolicy([{"cost": 0}], "minmax"),
        PreferencePolicy([{"cost": np.nan}], "minmax"),
        PreferencePolicy([{"cost": 1}], "bad"),
        PreferencePolicy([{"cost": 1}], "reference"),
        PreferencePolicy(
            [{"cost": 1}],
            "reference",
            reference_bounds={"cost": (1, 0), "accuracy": (0, 1)},
        ),
        PreferencePolicy(
            [{"cost": 1}],
            "reference",
            reference_bounds={"cost": (1, 1), "accuracy": (0, 1)},
        ),
        PreferencePolicy([{"cost": 1}], "none", equivalence={"cost": -1}),
        PreferencePolicy([{"cost": 1}], "none", paired_confidence=1),
    ],
)
def test_invalid_assumptions_are_rejected(policy: PreferencePolicy) -> None:
    with pytest.raises(
        ValueError,
        match=r"Preference|normalization|reference_bounds|Zero-range|Equivalence|paired",
    ):
        preference_sweep(_table(), _OBS, policy=policy)


def test_duplicate_or_incomplete_raw_identities_are_rejected() -> None:
    table = ResultsTable(
        [{"x": 0}, {"x": 0}],
        np.array([[1.0], [2.0]]),
        ["cost"],
        metadata=[{"design_point": 0, "rep": 0}, {"design_point": 0, "rep": 0}],
    )
    policy = PreferencePolicy([{"cost": 1}], "none")
    with pytest.raises(ValueError, match="Duplicate replicate"):
        preference_sweep(table, _OBS[:1], policy=policy)
    del table.metadata[1]["design_point"]
    with pytest.raises(ValueError, match="complete design_point/rep"):
        preference_sweep(table, _OBS[:1], policy=policy)
