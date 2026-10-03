"""Opt-in evaluation reuse with explicit revision and replicate identity."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ._checkpoint import _type_identity
from ._recovery import _grid_identity
from .protocols import TrialResult

if TYPE_CHECKING:
    from .protocols import Annotation, Observable, Scorer, Simulator


class EvaluationCache:
    """Persistent immutable evaluation evidence for explicitly identified runs."""

    def __init__(
        self,
        path: str | Path,
        *,
        revision: str,
        replicate_namespace: str,
        fidelity: str,
    ) -> None:
        """Create or reopen an evaluation cache.

        Args:
            path: SQLite cache file; compatible contexts may share one file.
            revision: Caller-managed simulator/scorer/data/annotation revision.
                Include every behavior input not identifiable from class code.
            replicate_namespace: Explicit randomness/seed experiment identity.
                Use a new namespace for genuinely independent replication.
            fidelity: Explicit simulation/evaluation fidelity identity.

        Raises:
            ValueError: If an identity is empty or the cache format is incompatible.
        """
        if any(
            not value.strip() for value in (revision, replicate_namespace, fidelity)
        ):
            msg = "revision, replicate_namespace and fidelity must be nonempty"
            raise ValueError(msg)
        self.path = Path(path)
        self.revision = revision
        self.replicate_namespace = replicate_namespace
        self.fidelity = fidelity
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path, timeout=30)) as connection, connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS cache_format (version INTEGER)"
            )
            stored = connection.execute("SELECT version FROM cache_format").fetchone()
            if stored is None:
                connection.execute("INSERT INTO cache_format VALUES (1)")
            elif stored[0] != 1:
                msg = "Incompatible evaluation cache format"
                raise ValueError(msg)
            connection.execute(
                "CREATE TABLE IF NOT EXISTS evaluations "
                "(key TEXT PRIMARY KEY, payload TEXT NOT NULL)"
            )

    def clear(self) -> None:
        """Invalidate all stored evaluations across every context in this file."""
        with closing(sqlite3.connect(self.path, timeout=30)) as connection, connection:
            connection.execute("DELETE FROM evaluations")


def _canonical(value: object) -> object:
    """Canonicalize configurations without collapsing opaque or container types.

    Returns:
        A typed JSON representation with stable dictionary ordering.

    Raises:
        ValueError: If a value has unsupported opaque behavior.
    """
    if isinstance(value, Enum):
        return {"enum": _type_identity(value), "value": _canonical(value.value)}
    if isinstance(value, np.generic):
        return {"numpy": str(value.dtype), "value": _canonical(value.item())}
    if type(value) in {type(None), bool, int, float, str}:
        return {"type": type(value).__name__, "value": value}
    if isinstance(value, (list, tuple)) and type(value) in {list, tuple}:
        return {"type": type(value).__name__, "value": [_canonical(v) for v in value]}
    if isinstance(value, dict) and type(value) is dict:
        return {
            "dict": sorted(
                (
                    json.dumps(_canonical(k), sort_keys=True, allow_nan=False),
                    _canonical(v),
                )
                for k, v in value.items()
            )
        }
    msg = "Cached configurations require supported JSON/scalar values"
    raise ValueError(msg)


@dataclass(frozen=True)
class _BoundCache:
    cache: EvaluationCache
    identity: str
    context: dict[str, str]

    def key(self, config: dict[str, Any], rep: int) -> str:
        value = json.dumps(
            {"context": self.identity, "config": _canonical(config), "rep": rep},
            sort_keys=True,
            allow_nan=False,
        )
        return hashlib.sha256(value.encode()).hexdigest()

    def load(self, config: dict[str, Any], rep: int) -> TrialResult | None:
        key = self.key(config, rep)
        with closing(sqlite3.connect(self.cache.path, timeout=30)) as connection:
            row = connection.execute(
                "SELECT payload FROM evaluations WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            return None
        payload = json.loads(row[0])
        return TrialResult(
            config,
            payload["scores"],
            payload["wall_seconds"],
            {
                "rep": rep,
                "cache_hit": True,
                "cache_key": key,
                "cache_context": dict(self.context),
            },
        )

    def save(self, config: dict[str, Any], rep: int, result: TrialResult) -> None:
        key = self.key(config, rep)
        payload = json.dumps(
            {"scores": result.scores, "wall_seconds": result.wall_seconds},
            sort_keys=True,
        )
        with (
            closing(sqlite3.connect(self.cache.path, timeout=30)) as connection,
            connection,
        ):
            connection.execute(
                "INSERT OR IGNORE INTO evaluations VALUES (?, ?)", (key, payload)
            )
            row = connection.execute(
                "SELECT payload FROM evaluations WHERE key = ?", (key,)
            ).fetchone()
            if json.dumps(json.loads(row[0])["scores"], sort_keys=True) != json.dumps(
                result.scores, sort_keys=True
            ):
                msg = (
                    "Conflicting scores for one cached evaluation identity; "
                    "change the replicate namespace/revision"
                )
                raise ValueError(msg)
        result.metadata.update({
            "cache_hit": False,
            "cache_key": key,
            "cache_context": dict(self.context),
        })


def _bind_cache(
    cache: EvaluationCache,
    world: Simulator,
    scorer: Scorer,
    observables: list[Observable],
    annotations: list[Annotation] | None,
) -> _BoundCache:
    identity = json.dumps(
        {
            "definition": _grid_identity(
                world, scorer, [], observables, annotations, 1, cache.revision
            ),
            "replicate_namespace": cache.replicate_namespace,
            "fidelity": cache.fidelity,
            "annotation_lookups": [
                _canonical(a.lookup)
                if isinstance(a.lookup, dict)
                else {
                    "module": getattr(a.lookup, "__module__", None),
                    "name": getattr(a.lookup, "__qualname__", None),
                }
                for a in (annotations or [])
            ],
        },
        sort_keys=True,
    )
    return _BoundCache(
        cache,
        identity,
        {
            "revision": cache.revision,
            "replicate_namespace": cache.replicate_namespace,
            "fidelity": cache.fidelity,
            "schema_digest": hashlib.sha256(identity.encode()).hexdigest(),
        },
    )
