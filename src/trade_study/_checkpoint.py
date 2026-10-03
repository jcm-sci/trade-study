"""Versioned structural identities for study checkpoints."""

from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from enum import Enum
from functools import partial
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .study import Study


def _type_identity(value: object) -> dict[str, str]:
    cls = type(value)
    identity = {"module": cls.__module__, "name": cls.__qualname__}
    try:
        source = inspect.getsource(cls)
    except (OSError, TypeError):
        return identity
    identity["source"] = hashlib.sha256(source.encode()).hexdigest()
    return identity


def _value(value: object, key: str | None) -> object:
    """Normalize structural values; require a key for opaque behavior.

    Returns:
        A JSON-compatible representation. Opaque values delegate to
        explicit-key validation in `_behavior_value`.
    """
    if isinstance(value, Enum):
        value = value.value
    elif isinstance(value, np.generic):
        value = value.item()
    elif is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _value(v, key) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_value(v, key) for v in value]
    return _behavior_value(value, key)


def _behavior_value(value: object, key: str | None) -> object:
    """Identify callable behavior or an explicitly versioned opaque value.

    Returns:
        A JSON-compatible callable or type identity.

    Raises:
        ValueError: If an opaque value has no explicit checkpoint key.
    """
    if isinstance(value, partial):
        return {
            "partial": _value(value.func, key),
            "args": _value(value.args, key),
            "kwargs": _value(value.keywords, key),
        }
    if inspect.isfunction(value):
        try:
            source = inspect.getsource(value)
        except (OSError, TypeError):
            source = None
        if source is not None:
            return {
                "module": value.__module__,
                "name": value.__qualname__,
                "source": hashlib.sha256(source.encode()).hexdigest(),
                "closure": _value(inspect.getclosurevars(value).nonlocals, key),
            }
    if key is None:
        msg = (
            f"Cannot identify checkpoint value of type {type(value).__qualname__}; "
            "set Study(checkpoint_key=...) to a caller-managed model revision"
        )
        raise ValueError(msg)
    return {"opaque_type": _type_identity(value)}


def _manifest(study: Study) -> dict[str, object]:
    """Return a JSON-normalized, versioned study definition.

    Raises:
        ValueError: If identity contains non-finite or unverifiable values.
    """
    key = study.checkpoint_key
    if key is not None and not key:
        msg = "checkpoint_key must be a non-empty revision string"
        raise ValueError(msg)
    definition = {
        "version": 1,
        "checkpoint_key": key,
        "phases": [phase.name for phase in study.phases],
        "observables": _value(study.observables, key),
        "factors": _value(study.factors, key),
        "annotations": _value(study.annotations, key),
        "definitions": [
            {
                "grid": _value(phase.grid, key),
                "filter": _value(phase.filter_fn, key),
                "n_trials": phase.n_trials,
                "n_reps": phase.n_reps,
                "world": _type_identity(
                    phase.world if phase.world is not None else study.world
                ),
                "scorer": _type_identity(
                    phase.scorer if phase.scorer is not None else study.scorer
                ),
            }
            for phase in study.phases
        ],
    }
    normalized: dict[str, object] = json.loads(
        json.dumps(definition, sort_keys=True, allow_nan=False)
    )
    return normalized
