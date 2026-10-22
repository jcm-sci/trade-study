"""Post-hoc sensitivity analysis from an already-collected ResultsTable.

``screen()`` (see :mod:`trade_study.design`) needs a live, re-runnable
simulator and a proper Saltelli/Morris sample design -- neither of which an
arbitrary already-collected :class:`~trade_study.protocols.ResultsTable`
has, so its Sobol/Morris indices can't be computed retroactively from
whatever points happen to be in the table.

:func:`sensitivity_from_table` bridges the gap: it fits a cheap surrogate
over the table (:func:`trade_study.fit_surrogate`) and runs ``screen()``'s
existing Sobol/Morris machinery against the surrogate's ``predict()``
instead of a fresh, expensive simulator evaluation. Sobol estimates variance
indices and Morris estimates elementary effects of the fitted surface, without
new simulator evaluations. Neither automatically measures discrete-factor effects.

The result is only as trustworthy as the surrogate it comes from. Check
:attr:`TableSensitivity.surrogate_cv_r2` before trusting the
indices for an observable with a poor cross-validated fit.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ._checkpoint import _value
from .design import Factor, FactorType, screen
from .protocols import ResultsTable
from .surrogate import _external_group_labels, fit_surrogate

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping, Sequence

    import numpy as np
    from numpy.typing import NDArray


@dataclass(frozen=True)
class TableSensitivity:
    """Post-hoc sensitivity indices computed via a table-fit surrogate.

    Attributes:
        importance: Mapping from observable name to an array of factor
            importances (mu_star for Morris, S1 for Sobol), one value per
            varying continuous factor, in ``screened_factors`` order.
        surrogate_cv_r2: Per-observable cross-validated R^2 of the
            surrogate the indices were computed from. A low value
            means the sensitivity indices reflect a poorly learned
            response surface, not necessarily the true system -- treat
            such observables' indices as unreliable. Grouped results stay
            unavailable/NaN when fewer than two groups survive.
        screened_factors: Ordered varying continuous input names.
        fixed_factors: Observed constants bound during prediction, including
            continuous factors made constant by conditioning.
        conditioning: Explicit requested discrete/categorical stratum.
        omitted_config_keys: Non-factor config fields excluded from the surface.
        metadata_keys: Explicitly declared non-factor bookkeeping fields.
        source_rows: Selected original row indices, aligned with validation_groups.
        sampling_factors: Factor definitions carrying screened bounds/log scales.
            Sampling is independent uniform in each declared linear/log coordinate.
        surrogate_cv_rmse: Actual held-out RMSE, in observable units.
        surrogate_row_cv_r2: Shuffled row R² retained alongside grouped metrics.
        surrogate_row_cv_rmse: Shuffled row RMSE.
        cv_group_by: Factor grouping names, or None for rows/external groups.
        cv_group_source: "rows", "configuration" or caller-declared "external".
        cv_group_labels: Validated external labels aligned with selected rows.
        validation_scheme: Shuffled rows, held-out designs or named factor groups.
            External groups are one explicit partition, not automatic crossed CV.
        validation_groups: Group ids aligned with source_rows, empty for row CV.
        cv_folds_requested: Requested maximum; effective folds depend on finite
            rows/groups for each observable and can be fewer or unavailable.
        support_ranges: Observed marginal continuous ranges per fitted observable.
            This does not establish support for joint sampled configurations.
    """

    importance: dict[str, NDArray[np.floating[Any]]]
    surrogate_cv_r2: dict[str, float]
    screened_factors: tuple[str, ...] = ()
    fixed_factors: dict[str, Any] = field(default_factory=dict)
    conditioning: dict[str, Any] = field(default_factory=dict)
    omitted_config_keys: tuple[str, ...] = ()
    metadata_keys: tuple[str, ...] = ()
    source_rows: tuple[int, ...] = ()
    sampling_factors: tuple[Factor, ...] = ()
    surrogate_cv_rmse: dict[str, float] = field(default_factory=dict)
    surrogate_row_cv_r2: dict[str, float] = field(default_factory=dict)
    surrogate_row_cv_rmse: dict[str, float] = field(default_factory=dict)
    cv_group_by: tuple[str, ...] | None = None
    cv_group_source: str = "rows"
    cv_group_labels: tuple[Hashable, ...] | None = None
    validation_scheme: str = "unreported"
    validation_groups: list[int] = field(default_factory=list)
    cv_folds_requested: int = 5
    support_ranges: dict[str, dict[str, tuple[float, float]]] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class _TableScope:
    table: ResultsTable
    continuous: list[Factor]
    fixed: dict[str, Any]
    rows: tuple[int, ...]
    omitted: tuple[str, ...]


def _key(value: object) -> str:
    """Reuse checkpoint value normalization for stable scalar/level comparison.

    Returns:
        A finite JSON key for a declared input value.
    """
    return json.dumps(_value(value, None), sort_keys=True, allow_nan=False)


def _scope_declarations(
    results: ResultsTable,
    factors: list[Factor],
    conditions: Mapping[str, Any],
    metadata_keys: Sequence[str],
) -> None:
    """Validate explicit factor, condition and bookkeeping declarations.

    Raises:
        ValueError: Declarations, rows or observed factor scope are invalid.
    """
    definitions = {factor.name: factor for factor in factors}
    keys = set().union(*(set(config) for config in results.configs))
    if len(definitions) != len(factors):
        msg = "Sensitivity factor names must be distinct"
        raise ValueError(msg)
    if (
        isinstance(metadata_keys, str)
        or len(set(metadata_keys)) != len(metadata_keys)
        or set(metadata_keys) & set(definitions)
        or set(metadata_keys) - keys
    ):
        msg = "metadata_keys must be distinct observed non-factor configuration keys"
        raise ValueError(msg)
    for name, value in conditions.items():
        if (
            name not in definitions
            or definitions[name].factor_type == FactorType.CONTINUOUS
        ):
            msg = "condition_on requires declared discrete/categorical factors"
            raise ValueError(msg)
        if value not in (definitions[name].levels or []):
            msg = f"Condition {name!r} is outside its declared levels"
            raise ValueError(msg)


def _selected_rows(
    results: ResultsTable, conditions: Mapping[str, Any]
) -> tuple[int, ...]:
    """Select existing rows without changing their validation or replicate identity.

    Returns:
        Ordered original row indices in a nonempty observed stratum.

    Raises:
        ValueError: Table layout or conditioning data are invalid.
    """
    if not results.configs:
        msg = "Sensitivity results table is empty"
        raise ValueError(msg)
    if (
        results.scores.shape != (len(results.configs), len(results.observable_names))
        or (results.metadata and len(results.metadata) != len(results.configs))
        or (
            results.annotations is not None
            and results.annotations.shape[0] != len(results.configs)
        )
    ):
        msg = "Sensitivity table score/config/annotation/metadata rows must align"
        raise ValueError(msg)
    if any(name not in config for name in conditions for config in results.configs):
        msg = "Conditioning factors must be present in every source row"
        raise ValueError(msg)
    rows = tuple(
        i
        for i, config in enumerate(results.configs)
        if all(config[name] == value for name, value in conditions.items())
    )
    if not rows:
        msg = "condition_on selected no observed rows"
        raise ValueError(msg)
    return rows


def _factor_scope(
    configs: list[dict[str, Any]], factors: list[Factor]
) -> tuple[list[Factor], dict[str, Any]]:
    """Hold observed constants fixed and reject varying noncontinuous inputs.

    Returns:
        Ordered varying continuous factors and observed constants.

    Raises:
        ValueError: Declared inputs are missing, invalid or unscreenable.
    """
    if any(factor.name not in config for factor in factors for config in configs):
        msg = "Every declared factor must be present in every selected row"
        raise ValueError(msg)
    fixed = {}
    continuous = []
    for factor in factors:
        values = [config[factor.name] for config in configs]
        if factor.factor_type != FactorType.CONTINUOUS and any(
            value not in (factor.levels or []) for value in values
        ):
            msg = f"Factor {factor.name!r} contains a value outside its declared levels"
            raise ValueError(msg)
        values = (
            list(map(float, values))
            if factor.factor_type == FactorType.CONTINUOUS
            else [
                next(level for level in (factor.levels or []) if level == value)
                for value in values
            ]
        )
        if len({_key(value) for value in values}) == 1:
            fixed[factor.name] = values[0]
        elif factor.factor_type == FactorType.CONTINUOUS:
            continuous.append(factor)
        else:
            msg = (
                f"Varying discrete/categorical factor {factor.name!r} cannot be "
                "omitted; select an observed stratum with condition_on"
            )
            raise ValueError(msg)
    if not continuous:
        msg = (
            "Screening requires at least one continuous factor with observed "
            "variation; use direct factorial or paired contrasts for discrete effects"
        )
        raise ValueError(msg)
    return continuous, fixed


def _condition_table(
    results: ResultsTable,
    factors: list[Factor],
    conditions: Mapping[str, Any],
    metadata_keys: Sequence[str],
) -> _TableScope:
    """Select an observed stratum and refuse unmodeled varying factors.

    Returns:
        Aligned selected table and explicit varying/fixed/omitted scope.

    Raises:
        ValueError: An undeclared varying config key could conceal an input.
    """
    _scope_declarations(results, factors, conditions, metadata_keys)
    rows = _selected_rows(results, conditions)
    configs = [dict(results.configs[i]) for i in rows]
    continuous, fixed = _factor_scope(configs, factors)
    omitted = tuple(
        sorted(
            set().union(*(set(config) for config in configs))
            - {f.name for f in factors}
        )
    )
    for name in omitted:
        if name not in metadata_keys and (
            any(name not in config for config in configs)
            or len({_key(config[name]) for config in configs}) > 1
        ):
            msg = (
                f"Varying unmodeled config key {name!r}; declare its factor or "
                "explicitly mark bookkeeping in metadata_keys"
            )
            raise ValueError(msg)
    return _TableScope(
        ResultsTable(
            configs,
            results.scores[list(rows)].copy(),
            list(results.observable_names),
            annotations=None
            if results.annotations is None
            else results.annotations[list(rows)].copy(),
            annotation_names=list(results.annotation_names),
            metadata=[dict(results.metadata[i]) for i in rows]
            if results.metadata
            else [],
        ),
        continuous,
        fixed,
        rows,
        omitted,
    )


def sensitivity_from_table(  # ruff: ignore[too-many-arguments]
    results: ResultsTable,
    factors: list[Factor],
    *,
    method: str = "sobol",
    surrogate_method: str = "rf",
    n_trajectories: int = 100,
    seed: int = 42,
    n_estimators: int = 200,
    warn_below_r2: float | None = 0.0,
    condition_on: Mapping[str, Any] | None = None,
    metadata_keys: Sequence[str] = (),
    cv_group_by: str | Sequence[str] | None = None,
    cv_groups: Sequence[object] | NDArray[Any] | None = None,
    cv_folds: int = 5,
) -> TableSensitivity:
    """Compute post-hoc Sobol/Morris sensitivity from a collected table.

    Only varying continuous factors are screened. Discrete/categorical factors
    must be observed constants, optionally after explicit conditioning. All
    constants are bound in surrogate predictions. Varying unmodeled config keys
    are refused unless explicitly declared bookkeeping through metadata_keys.
    This is conditional surface sensitivity under independent uniform sampling
    in declared linear/log coordinates, not sensitivity to discrete factors.

    Args:
        results: A :class:`~trade_study.protocols.ResultsTable` from a
            previous ``run_grid``/``Study``/etc. call.
        factors: All model factor definitions. At least one continuous factor
            must vary in the selected rows; observed constants are held fixed.
        method: ``"sobol"`` or ``"morris"``, forwarded to ``screen()``.
        surrogate_method: ``"rf"`` or ``"gp"``, forwarded to
            :func:`trade_study.fit_surrogate`.
        n_trajectories: Forwarded to ``screen()`` (Morris trajectory
            count, or Sobol base sample size).
        seed: Random seed for both the surrogate fit and ``screen()``.
        n_estimators: Forwarded to :func:`trade_study.fit_surrogate`
            (``rf`` only).
        warn_below_r2: Forwarded to :func:`trade_study.fit_surrogate`;
            warns if any observable's cross-validated R^2 is too low to
            trust its sensitivity indices. Pass ``None`` to disable.
        condition_on: Discrete/categorical values selecting an observed stratum.
            Remaining varying noncontinuous factors are refused.
        metadata_keys: Explicit non-factor config bookkeeping fields; excluded
            from prediction and reported in omitted_config_keys.
        cv_group_by: Forwarded configuration-factor grouping, including "design".
        cv_groups: Explicit dataset/replicate group labels aligned with every
            original results row, including rows excluded by conditioning.
            Mutually exclusive with cv_group_by. Values are validated before
            selection, then filtered by source_rows. No grouping is inferred.
        cv_folds: Requested CV maximum, forwarded to fit_surrogate. Must be >=2.

    Returns:
        A :class:`TableSensitivity` with importance indices and the
        surrogate validation, input scope, selected rows and marginal support.

    Raises:
        ValueError: Invalid scope/layout/conditioning, unmodeled variation or
            no varying continuous factor; also propagated from fit_surrogate.
    """
    if isinstance(cv_folds, bool) or not isinstance(cv_folds, int) or cv_folds < 2:
        msg = "cv_folds must be an integer >=2"
        raise ValueError(msg)
    if not any(f.factor_type == FactorType.CONTINUOUS for f in factors):
        msg = (
            "Screening requires at least one continuous factor; "
            "use direct factorial or paired contrasts for discrete effects"
        )
        raise ValueError(msg)
    if cv_groups is not None and cv_group_by is not None:
        msg = "cv_groups and cv_group_by are mutually exclusive"
        raise ValueError(msg)
    external = (
        None
        if cv_groups is None
        else _external_group_labels(cv_groups, len(results.configs))
    )
    conditions = dict(condition_on or {})
    scope = _condition_table(results, factors, conditions, metadata_keys)
    surrogate = fit_surrogate(
        scope.table,
        factors,
        method=surrogate_method,
        seed=seed,
        n_estimators=n_estimators,
        warn_below_r2=warn_below_r2,
        cv_group_by=cv_group_by,
        cv_folds=cv_folds,
        cv_groups=None if external is None else tuple(external[i] for i in scope.rows),
    )

    for name, ranges in surrogate.support_ranges.items():
        outside = [
            factor.name
            for factor in scope.continuous
            if factor.bounds is not None
            and (
                factor.bounds[0] < ranges[factor.name][0]
                or factor.bounds[1] > ranges[factor.name][1]
            )
        ]
        if outside:
            warnings.warn(
                f"Sensitivity sampling for {name!r} extends outside observed "
                f"marginal support for {outside}; "
                "inspect TableSensitivity.support_ranges",
                UserWarning,
                stacklevel=2,
            )

    def run_fn(cfg: dict[str, Any]) -> dict[str, float]:
        return surrogate.predict({**cfg, **scope.fixed}, warn_support=False)

    importance = screen(
        run_fn,
        scope.continuous,
        method=method,
        n_trajectories=n_trajectories,
        seed=seed,
    )
    return TableSensitivity(
        importance=importance,
        surrogate_cv_r2=surrogate.cv_r2,
        screened_factors=tuple(factor.name for factor in scope.continuous),
        fixed_factors=dict(scope.fixed),
        conditioning=conditions,
        omitted_config_keys=scope.omitted,
        metadata_keys=tuple(metadata_keys),
        source_rows=scope.rows,
        sampling_factors=tuple(scope.continuous),
        surrogate_cv_rmse=surrogate.cv_rmse,
        surrogate_row_cv_r2=surrogate.row_cv_r2,
        surrogate_row_cv_rmse=surrogate.row_cv_rmse,
        cv_group_by=surrogate.cv_group_by,
        cv_group_source=surrogate.cv_group_source,
        cv_group_labels=surrogate.cv_group_labels,
        validation_scheme="shuffled-rows"
        if surrogate.cv_group_source == "rows"
        else "held-out-external-groups"
        if surrogate.cv_group_source == "external"
        else "held-out-designs"
        if cv_group_by == "design"
        else "held-out-factor-groups",
        validation_groups=surrogate.validation_groups,
        cv_folds_requested=cv_folds,
        support_ranges=surrogate.support_ranges,
    )
