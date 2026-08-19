"""Post-hoc sensitivity analysis from an already-collected ResultsTable (#113).

``screen()`` (see :mod:`trade_study.design`) needs a live, re-runnable
simulator and a proper Saltelli/Morris sample design -- neither of which an
arbitrary already-collected :class:`~trade_study.protocols.ResultsTable`
has, so its Sobol/Morris indices can't be computed retroactively from
whatever points happen to be in the table.

:func:`sensitivity_from_table` bridges the gap: it fits a cheap surrogate
over the table (:func:`trade_study.fit_surrogate`) and runs ``screen()``'s
existing Sobol/Morris machinery against the surrogate's ``predict()``
instead of a fresh, expensive simulator evaluation. Because the surrogate
is cheap to query, this produces genuine variance-based sensitivity
indices -- which, unlike a marginal Spearman correlation, correctly detect
non-monotonic effects -- without any new simulator evaluations.

The result is only as trustworthy as the surrogate it comes from. Check
:attr:`TableSensitivity.surrogate_cv_r2` (see #114) before trusting the
indices for an observable with a poor cross-validated fit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .design import FactorType, screen
from .surrogate import fit_surrogate

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from .design import Factor
    from .protocols import ResultsTable


@dataclass(frozen=True)
class TableSensitivity:
    """Post-hoc sensitivity indices computed via a table-fit surrogate.

    Attributes:
        importance: Mapping from observable name to an array of factor
            importances (mu_star for Morris, S1 for Sobol), one value per
            continuous factor, in the same order as
            :func:`trade_study.screen` reports (i.e. continuous factors
            only, in the order they appear in the ``factors`` argument).
        surrogate_cv_r2: Per-observable cross-validated R^2 of the
            surrogate the indices were computed from (#114). A low value
            means the sensitivity indices reflect a poorly learned
            response surface, not necessarily the true system -- treat
            such observables' indices as unreliable.
    """

    importance: dict[str, NDArray[np.floating[Any]]]
    surrogate_cv_r2: dict[str, float]


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
) -> TableSensitivity:
    """Compute post-hoc Sobol/Morris sensitivity from a collected table.

    Only continuous factors are screened, matching ``screen()``'s own
    contract; the surrogate is fit on that same continuous subset, so any
    non-continuous keys present in ``results.configs`` (categorical
    factors, bookkeeping fields, etc.) are simply ignored rather than
    causing an encoding mismatch.

    Args:
        results: A :class:`~trade_study.protocols.ResultsTable` from a
            previous ``run_grid``/``Study``/etc. call.
        factors: Factor definitions to screen. Non-continuous factors are
            dropped (as in ``screen()``); at least one continuous factor
            must remain.
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

    Returns:
        A :class:`TableSensitivity` with importance indices and the
        surrogate's cross-validated accuracy per observable.

    Raises:
        ValueError: If ``factors`` has no continuous entries, or
            propagated from ``fit_surrogate()`` (e.g. an empty table).
    """
    continuous = [f for f in factors if f.factor_type == FactorType.CONTINUOUS]
    if not continuous:
        msg = "Screening requires at least one continuous factor"
        raise ValueError(msg)

    surrogate = fit_surrogate(
        results,
        continuous,
        method=surrogate_method,
        seed=seed,
        n_estimators=n_estimators,
        warn_below_r2=warn_below_r2,
    )

    def run_fn(cfg: dict[str, Any]) -> dict[str, float]:
        return surrogate.predict(cfg)

    importance = screen(
        run_fn,
        continuous,
        method=method,
        n_trajectories=n_trajectories,
        seed=seed,
    )
    return TableSensitivity(importance=importance, surrogate_cv_r2=surrogate.cv_r2)
