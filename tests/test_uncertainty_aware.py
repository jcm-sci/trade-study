"""Tests for uncertainty-aware recommendations and constraints (#115)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from trade_study import ResultsTable
from trade_study.design import Factor, FactorType
from trade_study.protocols import Constraint, Direction, Observable
from trade_study.regime import fit_regime_surrogate
from trade_study.session import AdaptiveSession
from trade_study.surrogate import fit_surrogate

FACTORS = [Factor("x", FactorType.CONTINUOUS, bounds=(0.0, 1.0))]


def _table(seed: int = 0) -> ResultsTable:
    rng = np.random.default_rng(seed)
    xs = rng.uniform(0.0, 1.0, 60)
    configs = [{"x": float(x), "n": 1.0} for x in xs]
    scores = ((xs - 0.4) ** 2 + 0.01 * rng.standard_normal(60)).reshape(-1, 1)
    return ResultsTable(configs=configs, scores=scores, observable_names=["loss"])


@pytest.mark.parametrize("method", ["gp", "rf"])
def test_spread_batch_is_non_negative_for_both_backends(method: str) -> None:
    table = _table()
    surrogate = fit_surrogate(
        ResultsTable(
            configs=[{"x": c["x"]} for c in table.configs],
            scores=table.scores,
            observable_names=table.observable_names,
        ),
        FACTORS,
        method=method,
    )

    spread = surrogate.spread_batch([{"x": 0.1}, {"x": 0.9}])["loss"]

    assert spread.shape == (2,)
    assert np.all(spread >= 0)


def test_risk_penalizes_uncertain_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    surrogate = fit_regime_surrogate(
        _table(),
        regime_factors=[Factor("n", FactorType.CONTINUOUS, bounds=(0.5, 1.5))],
        factors=FACTORS,
        method="rf",
    )
    candidates = [{"x": 0.1}, {"x": 0.2}]
    monkeypatch.setattr(
        surrogate,
        "predict_batch",
        lambda *_, **__: {"loss": np.array([0.10, 0.12])},
    )
    monkeypatch.setattr(
        surrogate.inner,
        "spread_batch",
        lambda *_, **__: {"loss": np.array([0.05, 0.001])},
    )

    plain = surrogate.recommend({"n": 1.0}, objective="loss", candidates=candidates)
    cautious = surrogate.recommend(
        {"n": 1.0}, objective="loss", candidates=candidates, risk=1.0
    )
    hopeful = surrogate.recommend(
        {"n": 1.0}, objective="loss", mode="max", candidates=candidates, risk=1.0
    )

    assert plain == {"x": 0.1}
    assert cautious == {"x": 0.2}
    assert hopeful == {"x": 0.2}
    with pytest.raises(ValueError, match="risk must be non-negative"):
        surrogate.recommend({"n": 1.0}, objective="loss", risk=-1.0)


def test_confident_constraint_uses_the_upper_bound() -> None:
    table = ResultsTable(
        configs=[{"x": 0}, {"x": 1}],
        scores=np.array([[0.04], [0.04]]),
        observable_names=["fpr"],
        metadata=[
            {"standard_error": {"fpr": 0.001}},
            {"standard_error": {"fpr": 0.02}},
        ],
    )

    point = Constraint("fpr", "fpr", "<=", 0.05)
    confident = Constraint("fpr", "fpr", "<=", 0.05, confidence=0.95)

    np.testing.assert_array_equal(table.feasible([point]), [True, True])
    np.testing.assert_array_equal(table.feasible([confident]), [True, False])


def test_confident_constraint_reads_aggregated_replicates() -> None:
    raw = ResultsTable(
        configs=[{"x": 0}] * 4 + [{"x": 1}] * 4,
        scores=np.array([[0.9], [1.1], [0.9], [1.1], [0.5], [1.5], [0.5], [1.5]]),
        observable_names=["power"],
        metadata=[{"design_point": i // 4} for i in range(8)],
    )
    table = raw.aggregate_replicates()

    floor = Constraint("power", "power", ">=", 0.8, confidence=0.9)

    np.testing.assert_array_equal(table.feasible([floor]), [True, False])


def test_confident_constraints_need_standard_errors() -> None:
    table = ResultsTable(
        configs=[{"x": 0}], scores=np.array([[0.1]]), observable_names=["fpr"]
    )

    with pytest.raises(ValueError, match="no standard error"):
        table.feasible([Constraint("fpr", "fpr", "<=", 0.05, confidence=0.95)])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"op": "<=", "confidence": 0.4}, "confidence must lie"),
        ({"op": "==", "confidence": 0.9}, "inequality operator"),
    ],
)
def test_invalid_confident_constraints(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Constraint("c", "fpr", threshold=0.05, **kwargs)


def test_session_constraints_use_the_confidence_bound() -> None:
    confident = Constraint("cheap", "cost", "<=", 0.5, confidence=0.95)
    session = AdaptiveSession(
        FACTORS, [Observable("cost", Direction.MINIMIZE)], constraints=[confident]
    )
    ((trial_id, _config),) = session.ask(1)

    session.tell(trial_id, {"cost": [0.4, 0.5, 0.45, 0.55]})

    meta = session.results().metadata[0]
    error = meta["standard_error"]["cost"]
    expected = 0.475 + 1.6448536269514722 * error - 0.5
    assert meta["constraints"] == pytest.approx([expected])


@pytest.mark.parametrize("op", ["<=", ">="])
@pytest.mark.parametrize("values", [0.4, [0.4, np.nan], [np.nan], [np.inf]])
def test_confident_session_rejects_unknown_uncertainty_and_allows_correction(
    op: str, values: float | list[float]
) -> None:
    session = AdaptiveSession(
        FACTORS,
        [Observable("cost", Direction.MINIMIZE)],
        constraints=[Constraint("cap", "cost", op, 0.5, confidence=0.95)],
    )
    ((trial_id, _config),) = session.ask(1)
    with pytest.raises(ValueError, match="finite"):
        session.tell(trial_id, {"cost": values})
    assert session.results().configs == []
    session.tell(trial_id, {"cost": [0.4, 0.45]})
    assert len(session.results().configs) == 1


@pytest.mark.parametrize("op", ["<=", ">="])
def test_session_requires_non_objective_constraint_score(op: str) -> None:
    session = AdaptiveSession(
        FACTORS,
        [Observable("loss", Direction.MINIMIZE)],
        constraints=[Constraint("cap", "cost", op, 0.5)],
    )
    ((trial_id, _config),) = session.ask(1)
    with pytest.raises(ValueError, match="Missing constraint score"):
        session.tell(trial_id, {"loss": 0.1})
    session.tell(trial_id, {"loss": 0.1, "cost": 0.4})
    assert session.results().metadata[0]["scores"]["cost"] == pytest.approx(0.4)


@pytest.mark.parametrize("error", [np.nan, np.inf, -0.1])
def test_table_rejects_invalid_confidence_standard_error(error: float) -> None:
    table = ResultsTable(
        configs=[{"x": 0}],
        scores=np.array([[0.4]]),
        observable_names=["cost"],
        metadata=[{"standard_error": {"cost": error}}],
    )
    with pytest.raises(ValueError, match="standard error"):
        table.feasible([Constraint("cap", "cost", "<=", 0.5, confidence=0.95)])


def test_confident_constraint_accepts_zero_variance_with_replicates() -> None:
    session = AdaptiveSession(
        FACTORS,
        [Observable("cost", Direction.MINIMIZE)],
        constraints=[Constraint("cap", "cost", "<=", 0.5, confidence=0.95)],
    )
    ((trial_id, _config),) = session.ask(1)
    session.tell(trial_id, {"cost": [0.4, 0.4]})
    assert session.results().metadata[0]["constraints"] == pytest.approx([-0.1])
