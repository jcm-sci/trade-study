"""Tests for paired comparisons under common random numbers (#133)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from trade_study import Direction, Observable, run_grid
from trade_study.paired import paired_difference, paired_rank
from trade_study.protocols import ResultsTable

N_REPS = 30


def _table(
    shifts: dict[str, float], *, n_reps: int = N_REPS, seed: int = 0
) -> ResultsTable:
    """Designs share one noise draw per replicate plus a small private noise.

    Returns:
        Per-replicate results with ``rep`` and ``design_point`` metadata.
    """
    rng = np.random.default_rng(seed)
    shared = rng.normal(0.0, 1.0, n_reps)
    configs, scores, metadata = [], [], []
    for point, (name, shift) in enumerate(shifts.items()):
        private = rng.normal(0.0, 0.05, n_reps)
        for rep in range(n_reps):
            configs.append({"design": name})
            scores.append([shift + shared[rep] + private[rep]])
            metadata.append({"rep": rep, "design_point": point})
    return ResultsTable(
        configs=configs,
        scores=np.array(scores),
        observable_names=["error"],
        metadata=metadata,
    )


def test_recovers_a_known_shift() -> None:
    table = _table({"a": 0.5, "b": 0.0})

    result = paired_difference(table, {"design": "a"}, {"design": "b"}, "error")

    assert result.n_pairs == N_REPS
    assert result.lower < 0.5 < result.upper
    assert result.mean == pytest.approx(0.5, abs=0.05)
    assert result.design_a == {"design": "a"}


def test_pairing_is_much_narrower_than_an_unpaired_interval() -> None:
    table = _table({"a": 0.5, "b": 0.0})
    a = table.scores[:N_REPS, 0]
    b = table.scores[N_REPS:, 0]
    unpaired_half = 1.96 * np.sqrt(a.var(ddof=1) / N_REPS + b.var(ddof=1) / N_REPS)

    result = paired_difference(table, 0, 1, "error")

    assert (result.upper - result.lower) / 2 < unpaired_half / 5


def test_t_interval_matches_the_textbook_formula() -> None:
    stats = pytest.importorskip("scipy.stats")
    table = _table({"a": 0.5, "b": 0.0})
    diff = table.scores[:N_REPS, 0] - table.scores[N_REPS:, 0]
    half = stats.t.ppf(0.975, N_REPS - 1) * diff.std(ddof=1) / np.sqrt(N_REPS)

    result = paired_difference(table, 0, 1, "error", method="t")

    assert result.lower == pytest.approx(diff.mean() - half)
    assert result.upper == pytest.approx(diff.mean() + half)


def test_bootstrap_is_reproducible_for_a_seed() -> None:
    table = _table({"a": 0.5, "b": 0.0})

    first = paired_difference(table, 0, 1, "error", seed=3)
    second = paired_difference(table, 0, 1, "error", seed=3)

    assert (first.lower, first.upper) == (second.lower, second.upper)


def test_mismatched_replicates_are_rejected() -> None:
    table = _table({"a": 0.5, "b": 0.0})
    table.metadata[0] = {"rep": 99, "design_point": 0}

    with pytest.raises(ValueError, match="different replicates"):
        paired_difference(table, 0, 1, "error")


def test_repeated_replicates_are_rejected() -> None:
    table = _table({"a": 0.5, "b": 0.0})
    table.metadata[1] = {"rep": 0, "design_point": 0}

    with pytest.raises(ValueError, match="repeats replicate"):
        paired_difference(table, 0, 1, "error")


def test_rows_without_rep_are_rejected() -> None:
    table = _table({"a": 0.5, "b": 0.0})
    table.metadata[0] = {"design_point": 0}

    with pytest.raises(ValueError, match="metadata\\['rep'\\]"):
        paired_difference(table, 0, 1, "error")


def test_ambiguous_and_missing_designs_are_rejected() -> None:
    table = _table({"a": 0.5, "b": 0.0})

    with pytest.raises(ValueError, match="2 distinct configs"):
        paired_difference(table, {}, 1, "error")
    with pytest.raises(ValueError, match="no rows match"):
        paired_difference(table, {"design": "z"}, 1, "error")


def test_non_finite_pairs_are_dropped() -> None:
    table = _table({"a": 0.5, "b": 0.0})
    table.scores[0, 0] = np.nan

    result = paired_difference(table, 0, 1, "error")

    assert result.n_pairs == N_REPS - 1


def test_unknown_method_and_observable_are_rejected() -> None:
    table = _table({"a": 0.5, "b": 0.0})

    with pytest.raises(ValueError, match="method"):
        paired_difference(table, 0, 1, "error", method="wilcoxon")
    with pytest.raises(KeyError, match="cost"):
        paired_difference(table, 0, 1, "cost")


def test_rank_orders_designs_against_the_reference() -> None:
    table = _table({"ref": 0.0, "worse": 0.3, "better": -0.4, "best": -0.8})

    ranked = paired_rank(table, "error", {"design": "ref"})

    assert [d.design_a["design"] for d in ranked] == ["best", "better", "worse"]
    assert all(d.design_b == {"design": "ref"} for d in ranked)
    maximized = paired_rank(table, "error", {"design": "ref"}, maximize=True)
    assert [d.design_a["design"] for d in maximized] == ["worse", "better", "best"]


def test_bonferroni_widens_each_interval() -> None:
    table = _table({"ref": 0.0, "a": 0.3, "b": -0.4, "c": -0.8})

    joint = paired_rank(table, "error", 0, method="t")
    single = paired_rank(table, "error", 0, method="t", adjust=None)

    for adjusted, plain in zip(joint, single, strict=True):
        assert adjusted.confidence == pytest.approx(1 - 0.05 / 3)
        assert adjusted.upper - adjusted.lower > plain.upper - plain.lower


def test_rank_rejects_unknown_adjustment() -> None:
    table = _table({"ref": 0.0, "a": 0.3})

    with pytest.raises(ValueError, match="adjust"):
        paired_rank(table, "error", 0, adjust="holm")


class _SharedNoiseSimulator:
    """Data depend on the replicate only, as under common random numbers."""

    def generate(self, config: dict[str, Any], *, rep: int = 0) -> tuple[float, float]:
        """Draw this replicate's noise.

        Returns:
            Tuple of (shift, noise).
        """
        return float(config["shift"]), float(np.random.default_rng(rep).normal())


class _ShiftScorer:
    def score(
        self, truth: float, observations: float, config: dict[str, Any]
    ) -> dict[str, float]:
        """Score is the shift plus the shared noise.

        Returns:
            Dict with ``error``.
        """
        del config
        return {"error": truth + observations}


def test_works_on_run_grid_output() -> None:
    table = run_grid(
        _SharedNoiseSimulator(),
        _ShiftScorer(),
        [{"shift": 0.2}, {"shift": 0.0}],
        [Observable("error", Direction.MINIMIZE)],
        n_reps=5,
    )

    result = paired_difference(table, {"shift": 0.2}, {"shift": 0.0}, "error")

    assert result.mean == pytest.approx(0.2)
    assert result.lower == pytest.approx(0.2)
    assert result.upper == pytest.approx(0.2)
