"""Tests for the batched ask/tell adaptive session (#132)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import optuna
import pytest

from trade_study.design import Factor, FactorType
from trade_study.protocols import Constraint, Direction, Observable
from trade_study.runner import run_adaptive
from trade_study.session import AdaptiveSession

if TYPE_CHECKING:
    from pathlib import Path

FACTORS = [
    Factor("x", FactorType.CONTINUOUS, bounds=(0.0, 1.0)),
    Factor("mode", FactorType.CATEGORICAL, levels=["a", "b"]),
]
OBSERVABLES = [
    Observable("error", Direction.MINIMIZE),
    Observable("cost", Direction.MINIMIZE, weight=2.0),
]


def _score(config: dict[str, Any]) -> dict[str, float]:
    bonus = 0.1 if config["mode"] == "b" else 0.0
    return {"error": abs(config["x"] - 0.3) + bonus, "cost": config["x"]}


class _World:
    def generate(self, config: dict[str, Any]) -> tuple[Any, Any]:
        return config, config


class _Scorer:
    def score(self, truth: Any, observations: Any, config: dict[str, Any]) -> dict:
        return _score(config)


def _previous_run_adaptive(n_trials: int, seed: int) -> list[dict[str, Any]]:
    """Reference: run_adaptive as implemented before the session refactor.

    Returns:
        The params of every trial, in order.
    """
    study = optuna.create_study(
        directions=["minimize", "minimize"],
        sampler=optuna.samplers.NSGAIISampler(seed=seed),
    )

    def objective(trial: optuna.trial.Trial) -> tuple[float, ...]:
        config = {
            "x": trial.suggest_float("x", 0.0, 1.0, log=False),
            "mode": trial.suggest_categorical("mode", ["a", "b"]),
        }
        scores = _score(config)
        return scores["error"] * 1.0, scores["cost"] * 2.0

    study.optimize(objective, n_trials=n_trials)
    return [trial.params for trial in study.trials]


def test_run_adaptive_reproduces_the_previous_implementation() -> None:
    result = run_adaptive(
        _World(), _Scorer(), FACTORS, OBSERVABLES, n_trials=30, seed=5
    )

    assert result.configs == _previous_run_adaptive(30, seed=5)


def test_in_process_ask_tell_matches_run_adaptive() -> None:
    session = AdaptiveSession(FACTORS, OBSERVABLES, seed=7)
    for _ in range(12):
        ((trial_id, config),) = session.ask(1)
        session.tell(trial_id, _score(config))

    expected = run_adaptive(
        _World(), _Scorer(), FACTORS, OBSERVABLES, n_trials=12, seed=7
    )
    np.testing.assert_allclose(session.results().scores, expected.scores)


def test_per_replicate_scores_keep_mean_standard_error_and_count() -> None:
    session = AdaptiveSession(FACTORS, OBSERVABLES, seed=1)
    ((trial_id, _config),) = session.ask(1)

    session.tell(trial_id, {"error": [0.1, 0.3, float("nan"), 0.2], "cost": 0.5})

    meta = session.results().metadata[0]
    assert meta["scores"]["error"] == pytest.approx(0.2)
    assert meta["standard_error"]["error"] == pytest.approx(0.1 / np.sqrt(3))
    assert meta["n_reps"] == {"error": 3, "cost": 1}
    np.testing.assert_allclose(session.results().scores[0], [0.2, 1.0])


def test_session_persists_between_processes(tmp_path: Path) -> None:
    path = tmp_path / "study.journal"
    first = AdaptiveSession(FACTORS, OBSERVABLES, path=path)
    proposals = first.ask(3)

    reopened = AdaptiveSession(FACTORS, OBSERVABLES, path=path)
    for trial_id, config in proposals:
        reopened.tell(trial_id, _score(config))

    results = AdaptiveSession(FACTORS, OBSERVABLES, path=path).results()
    assert sorted(m["trial"] for m in results.metadata) == [0, 1, 2]
    assert results.configs == [config for _, config in proposals]


def test_invalid_tells_are_rejected() -> None:
    session = AdaptiveSession(FACTORS, OBSERVABLES)
    ((trial_id, config),) = session.ask(1)

    with pytest.raises(ValueError, match="Missing objective"):
        session.tell(trial_id, {"error": 0.1})
    session.tell(trial_id, _score(config))
    with pytest.raises(ValueError, match="already told"):
        session.tell(trial_id, _score(config))
    with pytest.raises(ValueError, match="Unknown trial"):
        session.tell(99, _score(config))
    with pytest.raises(ValueError, match="n must be"):
        session.ask(0)


def test_constraint_values_reach_the_sampler(tmp_path: Path) -> None:
    path = tmp_path / "constrained.journal"
    cheap = Constraint("cheap", "cost", "<=", 0.5)
    session = AdaptiveSession(FACTORS, OBSERVABLES, constraints=[cheap], path=path)
    proposals = session.ask(4)
    for trial_id, config in proposals:
        session.tell(trial_id, _score(config))

    stored = optuna.load_study(
        study_name="trade-study-adaptive",
        storage=optuna.storages.JournalStorage(
            optuna.storages.journal.JournalFileBackend(str(path))
        ),
    )
    for trial, (_, config) in zip(stored.trials, proposals, strict=True):
        assert trial.system_attrs["constraints"] == pytest.approx([config["x"] - 0.5])
    reported = [m["constraints"] for m in session.results().metadata]
    assert reported == [pytest.approx([c["x"] - 0.5]) for _, c in proposals]


def test_inequality_constraints_are_rejected() -> None:
    with pytest.raises(ValueError, match="not supported"):
        AdaptiveSession(
            FACTORS, OBSERVABLES, constraints=[Constraint("odd", "cost", "!=", 0.0)]
        )
