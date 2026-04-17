"""Tests for viz module (issue #73)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from trade_study.protocols import Direction, ResultsTable
from trade_study.viz import (
    plot_calibration,
    plot_front,
    plot_parallel,
    plot_scores,
)

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")  # non-interactive backend for CI


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def two_obj_results() -> ResultsTable:
    """ResultsTable with 2 objectives (both minimize).

    Returns:
        A small results table with 6 configs.
    """
    scores = np.array([
        [1.0, 5.0],
        [2.0, 2.0],
        [5.0, 1.0],
        [3.0, 3.0],
        [4.0, 4.0],
        [3.5, 3.5],
    ])
    configs: list[dict[str, Any]] = [{"i": i} for i in range(len(scores))]
    return ResultsTable(
        configs=configs,
        scores=scores,
        observable_names=["obj1", "obj2"],
    )


@pytest.fixture
def three_obj_results() -> ResultsTable:
    """ResultsTable with 3 objectives (all minimize).

    Returns:
        A small results table with 6 configs and 3 objectives.
    """
    rng = np.random.default_rng(42)
    scores = rng.uniform(0, 10, size=(6, 3))
    configs: list[dict[str, Any]] = [{"i": i} for i in range(len(scores))]
    return ResultsTable(
        configs=configs,
        scores=scores,
        observable_names=["a", "b", "c"],
    )


@pytest.fixture
def four_obj_results() -> ResultsTable:
    """ResultsTable with 4 objectives (all minimize).

    Returns:
        A small results table with 8 configs and 4 objectives.
    """
    rng = np.random.default_rng(99)
    scores = rng.uniform(0, 10, size=(8, 4))
    configs: list[dict[str, Any]] = [{"i": i} for i in range(len(scores))]
    return ResultsTable(
        configs=configs,
        scores=scores,
        observable_names=["w", "x", "y", "z"],
    )


@pytest.fixture
def min2() -> list[Direction]:
    """Two minimize directions.

    Returns:
        List of two MINIMIZE directions.
    """
    return [Direction.MINIMIZE, Direction.MINIMIZE]


@pytest.fixture
def min3() -> list[Direction]:
    """Three minimize directions.

    Returns:
        List of three MINIMIZE directions.
    """
    return [Direction.MINIMIZE] * 3


@pytest.fixture
def min4() -> list[Direction]:
    """Four minimize directions.

    Returns:
        List of four MINIMIZE directions.
    """
    return [Direction.MINIMIZE] * 4


# ---------------------------------------------------------------------------
# plot_front
# ---------------------------------------------------------------------------


class TestPlotFront:
    """Tests for plot_front."""

    def test_2d_returns_fig_ax(
        self,
        two_obj_results: ResultsTable,
        min2: list[Direction],
    ):
        fig, ax = plot_front(two_obj_results, min2)
        assert fig is not None
        assert ax is not None

    def test_2d_with_existing_ax(
        self,
        two_obj_results: ResultsTable,
        min2: list[Direction],
    ):
        import matplotlib.pyplot as plt

        _, existing_ax = plt.subplots()
        fig, ax = plot_front(two_obj_results, min2, ax=existing_ax)
        assert ax is existing_ax
        plt.close(fig)

    def test_3d_returns_fig_ax(
        self,
        three_obj_results: ResultsTable,
        min3: list[Direction],
    ):
        fig, ax = plot_front(three_obj_results, min3)
        assert fig is not None
        assert ax is not None

    def test_pairs_returns_fig_axes(
        self,
        four_obj_results: ResultsTable,
        min4: list[Direction],
    ):
        fig, axes = plot_front(four_obj_results, min4)
        assert fig is not None
        assert axes.shape == (3, 3)

    def test_raises_on_single_objective(self):
        scores = np.array([[1.0], [2.0]])
        results = ResultsTable(
            configs=[{"i": 0}, {"i": 1}],
            scores=scores,
            observable_names=["only"],
        )
        with pytest.raises(ValueError, match="at least 2"):
            plot_front(results, [Direction.MINIMIZE])

    def test_custom_kwargs(
        self,
        two_obj_results: ResultsTable,
        min2: list[Direction],
    ):
        fig, _ax = plot_front(
            two_obj_results,
            min2,
            front_kw={"color": "red"},
            dominated_kw={"color": "blue"},
        )
        assert fig is not None


# ---------------------------------------------------------------------------
# plot_parallel
# ---------------------------------------------------------------------------


class TestPlotParallel:
    """Tests for plot_parallel."""

    def test_returns_fig_ax(
        self,
        two_obj_results: ResultsTable,
        min2: list[Direction],
    ):
        fig, ax = plot_parallel(two_obj_results, min2)
        assert fig is not None
        assert ax is not None

    def test_with_existing_ax(
        self,
        two_obj_results: ResultsTable,
        min2: list[Direction],
    ):
        import matplotlib.pyplot as plt

        _, existing_ax = plt.subplots()
        fig, ax = plot_parallel(two_obj_results, min2, ax=existing_ax)
        assert ax is existing_ax
        plt.close(fig)

    def test_custom_cmap(
        self,
        three_obj_results: ResultsTable,
        min3: list[Direction],
    ):
        fig, _ax = plot_parallel(three_obj_results, min3, cmap="plasma")
        assert fig is not None

    def test_maximize_direction(self):
        scores = np.array([[1.0, 5.0], [2.0, 2.0], [5.0, 1.0]])
        results = ResultsTable(
            configs=[{"i": i} for i in range(3)],
            scores=scores,
            observable_names=["cost", "quality"],
        )
        dirs = [Direction.MINIMIZE, Direction.MAXIMIZE]
        fig, _ax = plot_parallel(results, dirs)
        assert fig is not None


# ---------------------------------------------------------------------------
# plot_calibration
# ---------------------------------------------------------------------------


class TestPlotCalibration:
    """Tests for plot_calibration."""

    def test_returns_fig_ax(self):
        nominal = np.linspace(0.05, 0.95, 10)
        empirical = nominal + 0.02
        fig, ax = plot_calibration(nominal, empirical)
        assert fig is not None
        assert ax is not None

    def test_with_existing_ax(self):
        import matplotlib.pyplot as plt

        nominal = np.linspace(0.05, 0.95, 10)
        empirical = nominal
        _, existing_ax = plt.subplots()
        fig, ax = plot_calibration(nominal, empirical, ax=existing_ax)
        assert ax is existing_ax
        plt.close(fig)

    def test_ideal_calibration_line(self):
        nominal = np.linspace(0.0, 1.0, 5)
        empirical = nominal
        _fig, ax = plot_calibration(nominal, empirical)
        # Should have 2 lines: ideal diagonal and empirical
        assert len(ax.lines) == 2


# ---------------------------------------------------------------------------
# plot_scores
# ---------------------------------------------------------------------------


class TestPlotScores:
    """Tests for plot_scores."""

    def test_returns_fig_ax(self, two_obj_results: ResultsTable):
        fig, ax = plot_scores(two_obj_results, "obj1")
        assert fig is not None
        assert ax is not None

    def test_with_directions(
        self,
        two_obj_results: ResultsTable,
        min2: list[Direction],
    ):
        fig, _ax = plot_scores(two_obj_results, "obj2", min2)
        assert fig is not None

    def test_with_existing_ax(self, two_obj_results: ResultsTable):
        import matplotlib.pyplot as plt

        _, existing_ax = plt.subplots()
        fig, ax = plot_scores(two_obj_results, "obj1", ax=existing_ax)
        assert ax is existing_ax
        plt.close(fig)

    def test_raises_unknown_observable(self, two_obj_results: ResultsTable):
        with pytest.raises(ValueError, match="not_here"):
            plot_scores(two_obj_results, "not_here")
