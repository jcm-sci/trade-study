"""Schema and observation validation for importing adaptive evaluations."""

from __future__ import annotations

import json
from numbers import Real
from typing import TYPE_CHECKING, Any

import numpy as np

from ._checkpoint import _value
from .design import FactorType

if TYPE_CHECKING:
    import optuna

    from .design import Factor
    from .protocols import Constraint, Observable, ResultsTable


def _session_schema(
    factors: list[Factor],
    observables: list[Observable],
    constraints: list[Constraint],
    revision: str | None,
) -> str:
    return json.dumps(
        {
            "version": 1,
            "factors": _value(factors, revision),
            "observables": _value(observables, revision),
            "constraints": _value(constraints, revision),
            "revision": revision,
        },
        sort_keys=True,
        allow_nan=False,
    )


def _validate_config(config: dict[str, Any], factors: list[Factor]) -> dict[str, Any]:
    if set(config) != {f.name for f in factors}:
        msg = "Configuration must contain every session factor exactly"
        raise ValueError(msg)
    validated: dict[str, Any] = {}
    for factor in factors:
        value = config[factor.name]
        if factor.factor_type == FactorType.CONTINUOUS:
            value = float(value)
            if (
                factor.bounds is None
                or not factor.bounds[0] <= value <= factor.bounds[1]
            ):
                msg = f"Configuration outside bounds for {factor.name!r}"
                raise ValueError(msg)
        else:
            if value not in (factor.levels or []):
                msg = f"Configuration outside declared levels for {factor.name!r}"
                raise ValueError(msg)
            value = next(level for level in (factor.levels or []) if level == value)
        validated[factor.name] = value
    return validated


def _distributions(
    factors: list[Factor],
) -> dict[str, optuna.distributions.BaseDistribution]:
    import optuna as _optuna

    distributions: dict[str, optuna.distributions.BaseDistribution] = {}
    for factor in factors:
        if factor.factor_type == FactorType.CONTINUOUS and factor.bounds is not None:
            distributions[factor.name] = _optuna.distributions.FloatDistribution(
                *factor.bounds,
                log=factor.log_scale,
            )
        else:
            distributions[factor.name] = _optuna.distributions.CategoricalDistribution(
                factor.levels or [],
            )
    return distributions


def _summary(
    metadata: dict[str, Any],
    observables: list[Observable],
    constraints: list[Constraint],
    values: list[float],
) -> dict[str, Any]:
    scores = metadata.get("scores", {})
    counts = metadata.get("n_reps", {})
    errors = metadata.get("standard_error", {})
    required = {o.name for o in observables} | {c.observable for c in constraints}
    for name in required:
        count = counts.get(name, 0)
        mean = scores.get(name, np.nan)
        error = errors.get(name, np.nan)
        valid_count = (
            isinstance(count, int) and not isinstance(count, bool) and count >= 1
        )
        invalid_error = valid_count and count > 1 and (not _finite(error) or error < 0)
        if not valid_count or not _finite(mean) or invalid_error:
            msg = f"Invalid imported score/count/standard error for {name!r}"
            raise ValueError(msg)
    if not np.allclose(
        values, [scores[o.name] * o.weight for o in observables], rtol=1e-12, atol=1e-12
    ):
        msg = "Imported weighted objective values disagree with raw score metadata"
        raise ValueError(msg)
    for constraint in constraints:
        if constraint.confidence is not None and counts[constraint.observable] < 2:
            msg = "Imported confidence constraints need at least two replicates"
            raise ValueError(msg)
        constraint.bound(
            scores[constraint.observable], errors.get(constraint.observable, np.nan)
        )
    return {
        "scores": dict(scores),
        "n_reps": dict(counts),
        "standard_error": dict(errors),
    }


def _finite(value: object) -> bool:
    return isinstance(value, Real) and bool(np.isfinite(float(value)))


def _import_trials(
    results: ResultsTable,
    factors: list[Factor],
    observables: list[Observable],
    constraints: list[Constraint],
    schema: str,
) -> list[optuna.trial.FrozenTrial]:
    import optuna as _optuna

    if (
        results.observable_names != [o.name for o in observables]
        or results.scores.shape != (len(results.configs), len(observables))
        or len(results.metadata) != len(results.configs)
    ):
        msg = "Imported results have an incompatible observable or row schema"
        raise ValueError(msg)
    templates = []
    for config, values, meta in zip(
        results.configs, results.scores, results.metadata, strict=True
    ):
        identity = meta.get("evaluation_id")
        if (
            meta.get("session_schema") != schema
            or not isinstance(identity, str)
            or not identity.strip()
        ):
            msg = (
                "Imported results require matching session schema "
                "and evaluation provenance"
            )
            raise ValueError(msg)
        summary = _summary(meta, observables, constraints, values.tolist())
        summary.update({
            "evaluation_id": meta["evaluation_id"],
            "provenance": [
                *meta.get("provenance", []),
                {"session_id": meta.get("session_id"), "trial": meta.get("trial")},
            ],
        })
        templates.append(
            _optuna.trial.create_trial(
                state=_optuna.trial.TrialState.COMPLETE,
                values=values.tolist(),
                params=_validate_config(config, factors),
                distributions=_distributions(factors),
                user_attrs=summary,
            )
        )
    return templates


def _fingerprint(trial: optuna.trial.FrozenTrial) -> str:
    return json.dumps(
        {
            "params": trial.params,
            "values": trial.values,
            "summary": {
                k: trial.user_attrs.get(k)
                for k in ("scores", "n_reps", "standard_error")
            },
        },
        sort_keys=True,
    )
