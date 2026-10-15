"""Ordinary replicate summaries must compose with strict result artifacts (#184)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from trade_study import (
    Direction,
    Observable,
    ResultsTable,
    load_result_bundle,
    save_result_bundle,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_finite_aggregate_round_trips_through_versioned_bundle(tmp_path: Path) -> None:
    raw = ResultsTable(
        configs=[{"design": "a"}, {"design": "a"}],
        scores=np.array([[0.0, 2.0], [1.0, 4.0]]),
        observable_names=["recovered", "cost"],
        metadata=[{"design_point": 0, "rep": rep} for rep in range(2)],
    )
    summary = raw.aggregate_replicates()
    save_result_bundle(
        summary,
        tmp_path / "bundle",
        observables=[
            Observable("recovered", Direction.MAXIMIZE),
            Observable("cost", Direction.MINIMIZE),
        ],
    )
    restored = load_result_bundle(tmp_path / "bundle").results
    assert restored.configs == [{"design": "a"}]
    assert restored.metadata == [
        {"design_point": 0, "n_reps": 2, "score_std": {"recovered": 0.5, "cost": 1.0}},
    ]
    assert restored.metadata == summary.metadata
    np.testing.assert_array_equal(restored.scores, [[0.5, 3.0]])
