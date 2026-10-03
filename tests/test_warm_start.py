"""Schema-validated queueing, observation imports, and persistent provenance."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np
import optuna
import pytest

from trade_study import (
    AdaptiveSession,
    Constraint,
    Direction,
    Factor,
    FactorType,
    Observable,
    ResultsTable,
    load_results,
    save_results,
)

if TYPE_CHECKING:
    from pathlib import Path

_FACTORS = [
    Factor("x", FactorType.CONTINUOUS, bounds=(0.1, 10), log_scale=True),
    Factor("mode", FactorType.CATEGORICAL, levels=["a", "b"]),
    Factor("size", FactorType.DISCRETE, levels=[1, 2, 4]),
]
_OBS = [
    Observable("loss", Direction.MINIMIZE, weight=2),
    Observable("reward", Direction.MAXIMIZE),
]
_CONFIG = {"x": 1.0, "mode": "b", "size": 2}
_CONSTRAINTS = [Constraint("safe", "aux", "<=", 5)]


def _source() -> AdaptiveSession:
    session = AdaptiveSession(
        _FACTORS, _OBS, constraints=_CONSTRAINTS, revision="model-v1-data-v2"
    )
    session.enqueue(_CONFIG)
    ((trial, _),) = session.ask()
    session.tell(trial, {"loss": [1.0, 2.0], "reward": [2.0, 4.0], "aux": [1.0, 3.0]})
    return session


def _destination(path: Path | None = None) -> AdaptiveSession:
    return AdaptiveSession(
        _FACTORS,
        _OBS,
        constraints=_CONSTRAINTS,
        revision="model-v1-data-v2",
        path=path,
    )


def test_queued_mixed_log_configuration_and_seeded_proposals(tmp_path: Path) -> None:
    source = _source()
    first, second = _destination(), _destination()
    for session in (first, second):
        assert session.warm_start(source) == [0]
        session.enqueue(_CONFIG)
        assert session.trials("waiting")[0].config == _CONFIG
    assert first.ask(5) == second.ask(5)
    assert first.trials("pending")[0].config == _CONFIG
    path = tmp_path / "queued.journal"
    persisted = _destination(path)
    persisted.enqueue(_CONFIG)
    ((_, config),) = _destination(path).ask()
    assert config == _CONFIG


def test_saved_observations_import_once_even_through_another_session(
    tmp_path: Path,
) -> None:
    source = _source()
    save_results(source.results(), tmp_path / "table")
    table = load_results(tmp_path / "table")
    path = tmp_path / "destination.journal"
    destination = _destination(path)
    assert destination.warm_start(table) == [0]
    np.testing.assert_array_equal(destination.results().scores, source.results().scores)
    meta = destination.results().metadata[0]
    assert meta["n_reps"] == {"loss": 2, "reward": 2, "aux": 2}
    assert meta["scores"] == {"loss": 1.5, "reward": 3.0, "aux": 2.0}
    assert (
        meta["provenance"][0]["session_id"]
        == source.results().metadata[0]["session_id"]
    )
    assert destination.warm_start(source) == []
    assert _destination(path).warm_start(table) == []
    intermediate = _destination()
    intermediate.warm_start(source)
    assert _destination(path).warm_start(intermediate) == []
    assert source.warm_start(source) == []


@pytest.mark.parametrize("change", ["revision", "weight", "factor", "constraint"])
def test_schema_mismatch_is_rejected(change: str) -> None:
    source = _source()
    factors = _FACTORS[:-1] if change == "factor" else _FACTORS
    obs = (
        [Observable("loss", Direction.MINIMIZE), _OBS[1]]
        if change == "weight"
        else _OBS
    )
    constraints = (
        [Constraint("safe", "aux", "<=", 4)] if change == "constraint" else _CONSTRAINTS
    )
    revision = "model-v2-data-v2" if change == "revision" else "model-v1-data-v2"
    destination = AdaptiveSession(
        factors, obs, constraints=constraints, revision=revision
    )
    with pytest.raises(ValueError, match="matching session schema"):
        destination.warm_start(source)
    assert destination.results().configs == []


@pytest.mark.parametrize(
    "change", ["schema", "count", "error", "weighted", "config", "provenance"]
)
def test_malformed_import_is_rejected_without_adding_trials(change: str) -> None:
    table = copy.deepcopy(_source().results())
    if change == "schema":
        table.observable_names.reverse()
    elif change == "count":
        table.metadata[0]["n_reps"]["loss"] = "two"
    elif change == "error":
        table.metadata[0]["standard_error"]["loss"] = float("nan")
    elif change == "weighted":
        table.scores[0, 0] = 100
    elif change == "config":
        table.configs[0]["x"] = 100
    else:
        del table.metadata[0]["evaluation_id"]
    destination = _destination()
    with pytest.raises(ValueError, match=r"Imported|imported|Configuration"):
        destination.warm_start(table)
    assert destination.trials() == []


def test_conflicting_existing_evaluation_is_rejected() -> None:
    table = _source().results()
    destination = _destination()
    destination.warm_start(table)
    table.scores[0, 0] = 4
    table.metadata[0]["scores"]["loss"] = 2
    with pytest.raises(ValueError, match="Conflicting imported evaluation"):
        destination.warm_start(table)
    np.testing.assert_array_equal(destination.results().scores, [[3, 3]])


def test_import_requires_revision_and_source_schema() -> None:
    destination = AdaptiveSession(_FACTORS, _OBS)
    with pytest.raises(ValueError, match="explicit model/data revision"):
        destination.warm_start(_source())
    grid_table = ResultsTable([_CONFIG], np.array([[1.0, 2.0]]), ["loss", "reward"])
    with pytest.raises(ValueError, match="row schema"):
        _destination().warm_start(grid_table)


@pytest.mark.parametrize(
    "config",
    [
        {"x": 1.0},
        {**_CONFIG, "x": 0},
        {**_CONFIG, "mode": "other"},
        {**_CONFIG, "size": 3},
        {**_CONFIG, "extra": 1},
    ],
)
def test_invalid_queued_configuration_is_rejected(config: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="Configuration"):
        _destination().enqueue(config)


def test_reopening_validates_schema_and_refuses_legacy_storage(tmp_path: Path) -> None:
    path = tmp_path / "schema.journal"
    _destination(path).ask()
    with pytest.raises(ValueError, match="session schema"):
        AdaptiveSession(
            _FACTORS, _OBS, constraints=_CONSTRAINTS, revision="different", path=path
        )
    legacy_path = tmp_path / "legacy.journal"
    legacy = optuna.create_study(
        study_name="trade-study-adaptive",
        storage=optuna.storages.JournalStorage(
            optuna.storages.journal.JournalFileBackend(str(legacy_path))
        ),
    )
    legacy.ask()
    with pytest.raises(ValueError, match="legacy session schema"):
        _destination(legacy_path)
    with pytest.raises(ValueError, match="nonempty"):
        AdaptiveSession(_FACTORS, _OBS, revision=" ")
