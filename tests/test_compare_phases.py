"""Tests for Study.compare_phases (#81)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from trade_study.protocols import Direction, Observable
from trade_study.study import Phase, Study


class _ConfigSimulator:
    """Pass the config through as truth and observations."""

    def generate(self, config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return the config twice.

        Returns:
            Tuple of (config, config).
        """
        return config, config


class _AlphaScorer:
    """error = |alpha - 0.5|, cost = 10 alpha; NaN error when alpha exceeds 1."""

    def score(
        self, truth: Any, observations: Any, config: dict[str, Any]
    ) -> dict[str, float]:
        """Score one config.

        Returns:
            Dict with ``error`` and ``cost``.
        """
        del truth, observations
        a = float(config["alpha"])
        error = math.nan if a > 1.0 else abs(a - 0.5)
        return {"error": error, "cost": a * 10.0}


OBSERVABLES = [
    Observable("error", Direction.MINIMIZE),
    Observable("cost", Direction.MINIMIZE),
]


def _study(*grids: list[float], n_reps: int = 1) -> Study:
    phases = [
        Phase(name=f"p{i}", grid=[{"alpha": a} for a in grid], n_reps=n_reps)
        for i, grid in enumerate(grids)
    ]
    study = Study(
        world=_ConfigSimulator(),
        scorer=_AlphaScorer(),
        observables=OBSERVABLES,
        phases=phases,
    )
    study.run()
    return study


def test_improving_phase_has_positive_gain_and_no_loss() -> None:
    study = _study([0.0, 1.0], [0.0, 0.5])

    first, second = study.compare_phases(ref_point=np.array([1.0, 20.0]))

    assert first["igd_plus_gain"] is None
    assert first["igd_plus_loss"] is None
    assert first["n_front"] == 1
    assert first["hypervolume"] == pytest.approx(10.0)
    assert second["n_front"] == 2
    assert second["hypervolume"] == pytest.approx(17.5)
    assert second["igd_plus_gain"] == pytest.approx(0.25)
    assert second["igd_plus_loss"] == pytest.approx(0.0)


def test_identical_fronts_have_no_gain_or_loss() -> None:
    study = _study([0.0, 0.5], [0.5, 0.0])

    _first, second = study.compare_phases()

    assert second["igd_plus_gain"] == pytest.approx(0.0)
    assert second["igd_plus_loss"] == pytest.approx(0.0)


def test_best_values_follow_each_direction() -> None:
    study = _study([0.0, 1.0], [0.25, 0.75])

    first, second = study.compare_phases()

    assert first["best"] == {"error": pytest.approx(0.5), "cost": pytest.approx(0.0)}
    assert second["best"] == {"error": pytest.approx(0.25), "cost": pytest.approx(2.5)}


def test_default_reference_pads_the_worst_front_values() -> None:
    study = _study([0.0, 1.0], [0.0, 0.5])

    default = study.compare_phases()
    explicit = study.compare_phases(ref_point=np.array([0.55, 5.5]))

    for auto, given in zip(default, explicit, strict=True):
        assert auto["hypervolume"] == pytest.approx(given["hypervolume"])
        assert auto["hypervolume"] > 0


def test_default_reference_handles_maximized_observables() -> None:
    study = _study([0.5, 0.75, 1.0])
    study.observables = [
        Observable("error", Direction.MINIMIZE),
        Observable("cost", Direction.MAXIMIZE),
    ]

    (row,) = study.compare_phases()

    assert row["n_front"] == 3
    assert row["best"]["cost"] == pytest.approx(10.0)
    assert row["hypervolume"] > 0


def test_rows_with_non_finite_scores_are_left_out_of_the_front() -> None:
    study = _study([0.0, 0.5, 1.5])

    (row,) = study.compare_phases()

    assert row["n_trials"] == 3
    assert row["n_front"] == 2
    assert row["best"]["cost"] == pytest.approx(0.0)


def test_replicated_phases_are_compared_over_design_points() -> None:
    study = _study([0.0, 1.0], [0.0, 0.5], n_reps=3)

    first, second = study.compare_phases(ref_point=np.array([1.0, 20.0]))

    assert first["n_trials"] == 2
    assert second["n_trials"] == 2
    assert second["hypervolume"] == pytest.approx(17.5)


def test_phases_without_results_are_skipped() -> None:
    study = _study([0.0, 1.0])
    study.phases.append(Phase(name="pending", grid=[{"alpha": 0.5}]))

    rows = study.compare_phases()

    assert [row["phase"] for row in rows] == ["p0"]
