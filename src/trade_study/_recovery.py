"""Transactional, per-evaluation grid recovery storage."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ._checkpoint import _type_identity, _value
from .protocols import TrialResult

if TYPE_CHECKING:
    from .protocols import Annotation, Observable, Scorer, Simulator


def _grid_identity(
    world: Simulator,
    scorer: Scorer,
    grid: list[dict[str, Any]],
    observables: list[Observable],
    annotations: list[Annotation] | None,
    n_reps: int,
    key: str | None,
) -> str:
    """Identify evaluation behavior and ordered task identities.

    Returns:
        Canonical JSON identifying this grid run.

    Raises:
        ValueError: If the revision is empty or definitions are unverifiable.
    """
    if key is not None and not key.strip():
        msg = "checkpoint_key must be a nonempty revision string"
        raise ValueError(msg)
    return json.dumps(
        {
            "version": 1,
            "world": _type_identity(world),
            "scorer": _type_identity(scorer),
            "grid": _value(grid, key),
            "observables": _value(observables, key),
            "annotations": _value(annotations, key),
            "n_reps": n_reps,
            "checkpoint_key": key,
        },
        sort_keys=True,
        allow_nan=False,
    )


class _GridLedger:
    """Keep completed evaluations even when their containing phase fails."""

    def __init__(self, path: str | Path, identity: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path, timeout=30)) as connection, connection:
            connection.execute("CREATE TABLE IF NOT EXISTS identity (value TEXT)")
            stored = connection.execute("SELECT value FROM identity").fetchone()
            if stored is None:
                connection.execute("INSERT INTO identity VALUES (?)", (identity,))
            elif stored[0] != identity:
                msg = "Incompatible grid checkpoint; use a new checkpoint path"
                raise ValueError(msg)
            connection.execute(
                "CREATE TABLE IF NOT EXISTS trials ("
                "design_point INTEGER, rep INTEGER, attempts INTEGER NOT NULL, "
                "payload TEXT, last_error TEXT, PRIMARY KEY (design_point, rep))"
            )

    def load(
        self, design_point: int, rep: int, config: dict[str, Any]
    ) -> TrialResult | None:
        with closing(sqlite3.connect(self.path, timeout=30)) as connection:
            row = connection.execute(
                "SELECT payload FROM trials WHERE design_point = ? AND rep = ?",
                (design_point, rep),
            ).fetchone()
        if row is None or row[0] is None:
            return None
        payload = json.loads(row[0])
        return TrialResult(
            config=config,
            scores=payload["scores"],
            wall_seconds=payload["wall_seconds"],
            metadata={**payload["metadata"], "recovered": True},
        )

    def record(
        self, design_point: int, rep: int, result: TrialResult | None, error: str | None
    ) -> int:
        with closing(sqlite3.connect(self.path, timeout=30)) as connection, connection:
            connection.execute(
                "INSERT INTO trials VALUES (?, ?, 1, NULL, ?) "
                "ON CONFLICT (design_point, rep) DO UPDATE SET "
                "attempts = attempts + 1, last_error = excluded.last_error",
                (design_point, rep, error),
            )
            row = connection.execute(
                "SELECT attempts FROM trials WHERE design_point = ? AND rep = ?",
                (design_point, rep),
            ).fetchone()
            attempts = int(row[0])
            if result is not None:
                result.metadata["attempts"] = attempts
                payload = json.dumps({
                    "scores": result.scores,
                    "wall_seconds": result.wall_seconds,
                    "metadata": result.metadata,
                })
                connection.execute(
                    "UPDATE trials SET payload = ? WHERE design_point = ? AND rep = ?",
                    (payload, design_point, rep),
                )
            return attempts
