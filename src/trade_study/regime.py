"""Regime-conditional surrogate models (#105).

Builds on :mod:`trade_study.surrogate` to share information across
regime buckets. Regime descriptors (e.g. ``n_samples``, ``noise``) are
treated as additional input dimensions of a single surrogate over
``regime_factors + factors``, so the model can interpolate factor
recommendations across continuous regime axes instead of relying on
hard buckets.

Typical use:

.. code-block:: python

    surrogate = fit_regime_surrogate(
        results,
        regime_factors=[
            Factor(
                "n_samples", FactorType.CONTINUOUS, bounds=(1_000, 10_000)
            )
        ],
        factors=[Factor("lr", FactorType.CONTINUOUS, bounds=(1e-4, 1e-1))],
        method="gp",
    )
    best = surrogate.recommend({"n_samples": 2200}, objective="val_loss")

Optional dependency: install via the ``trade-study[surrogate]`` extra.
"""

from __future__ import annotations

import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .design import Factor, FactorType, build_grid
from .protocols import Direction
from .runner import run_adaptive
from .surrogate import SurrogateModel, fit_surrogate

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from numpy.typing import NDArray

    from .protocols import Observable, ResultsTable, Scorer, Simulator


_SUPPORTED_MODES: frozenset[str] = frozenset({"min", "max"})


def _merge(regime: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    """Merge a regime dict into a factor config dict (regime keys win).

    Args:
        regime: Mapping of regime-feature names to values.
        cfg: Mapping of design-factor names to values.

    Returns:
        New dict containing both sets of keys.
    """
    out = dict(cfg)
    out.update(regime)
    return out


@dataclass
class RegimeSurrogate:
    """Surrogate that conditions on regime features.

    Wraps a single :class:`SurrogateModel` fit over the union of regime
    descriptors and design factors. Use :func:`fit_regime_surrogate` to
    construct one.

    Attributes:
        inner: The underlying :class:`SurrogateModel` over the joint
            ``regime_factors + factors`` input space.
        regime_factors: Factors describing the regime (additional input
            dimensions of the surrogate).
        factors: Tunable design factors that are optimized at a given
            regime by :meth:`recommend`.
    """

    inner: SurrogateModel
    regime_factors: list[Factor]
    factors: list[Factor]

    @property
    def cv_r2(self) -> dict[str, float]:
        """Per-observable held-out cross-validated R^2 (#114).

        Returns:
            Mapping from observable name to cross-validated R^2, from the
            underlying :attr:`inner` surrogate.
        """
        return self.inner.cv_r2

    @property
    def cv_rmse(self) -> dict[str, float]:
        """Per-observable held-out cross-validated RMSE (#114).

        Returns:
            Mapping from observable name to cross-validated RMSE, from
            the underlying :attr:`inner` surrogate.
        """
        return self.inner.cv_rmse

    def predict(
        self,
        regime: dict[str, Any],
        config: dict[str, Any],
    ) -> dict[str, float]:
        """Predict observables at a regime + config pair.

        Args:
            regime: Mapping of regime-feature names to values.
            config: Mapping of design-factor names to values.

        Returns:
            Mapping from observable name to predicted scalar.
        """
        return self.inner.predict(_merge(regime, config))

    def predict_batch(
        self,
        regime: dict[str, Any],
        configs: Sequence[dict[str, Any]],
    ) -> dict[str, NDArray[np.float64]]:
        """Predict observables for a batch of configs at one regime.

        Args:
            regime: Mapping of regime-feature names to values.
            configs: Sequence of design-factor configs to score at
                ``regime``.

        Returns:
            Mapping from observable name to a length-``len(configs)``
            array of predictions.
        """
        merged = [_merge(regime, c) for c in configs]
        return self.inner.predict_batch(merged)

    def uncertainty(
        self,
        regime: dict[str, Any],
        config: dict[str, Any],
    ) -> dict[str, float]:
        """Predictive standard deviation per observable (GP only).

        Args:
            regime: Mapping of regime-feature names to values.
            config: Mapping of design-factor names to values.

        Returns:
            Mapping from observable name to predictive standard deviation.
            Propagates :class:`NotImplementedError` from the underlying
            surrogate when the backend does not expose calibrated
            uncertainties (non-GP backends).
        """
        return self.inner.uncertainty(_merge(regime, config))

    def recommend(
        self,
        regime: dict[str, Any],
        *,
        objective: str,
        mode: str = "min",
        n_candidates: int = 512,
        seed: int = 0,
        candidates: Sequence[dict[str, Any]] | None = None,
        warn_below_r2: float | None = 0.0,
    ) -> dict[str, Any]:
        """Recommend a design-factor config at a query regime.

        Samples ``n_candidates`` configs from the design-factor space via
        a scrambled Sobol' sequence and returns the one whose surrogate
        prediction for ``objective`` is best under ``mode``.

        Args:
            regime: Mapping of regime-feature names to values.
            objective: Name of the observable to optimize. Must be one
                of ``self.inner.observable_names``.
            mode: ``"min"`` or ``"max"``.
            n_candidates: Number of design-space samples to evaluate.
                Ignored when ``candidates`` is provided.
            seed: Seed for the Sobol' sampler.
            candidates: Optional explicit list of design-factor configs
                to score; if given, overrides ``n_candidates``.
            warn_below_r2: Warn if ``objective``'s cross-validated R^2
                (#114) is below this threshold, so a caller optimizing
                against a poorly-fit surrogate gets a signal right at the
                point of use, not just buried in fit-time logs. Pass
                ``None`` to disable.

        Returns:
            The candidate config (a copy) achieving the best predicted
            ``objective`` under ``mode``.

        Raises:
            ValueError: If ``objective`` is not a fitted observable, if
                ``mode`` is not ``"min"`` or ``"max"``, or if there are
                no candidates to score.
        """
        if mode not in _SUPPORTED_MODES:
            msg = f"mode must be one of {sorted(_SUPPORTED_MODES)}; got {mode!r}"
            raise ValueError(msg)
        if objective not in self.inner.observable_names:
            msg = (
                f"objective {objective!r} is not a fitted observable; "
                f"available: {self.inner.observable_names}"
            )
            raise ValueError(msg)
        r2 = self.cv_r2.get(objective, float("nan"))
        if warn_below_r2 is not None and np.isfinite(r2) and r2 < warn_below_r2:
            warnings.warn(
                f"recommend: objective {objective!r} has cross-validated "
                f"R^2={r2:.3f} (< {warn_below_r2}); this recommendation may "
                f"not be trustworthy.",
                UserWarning,
                stacklevel=2,
            )
        pool = (
            list(candidates)
            if candidates is not None
            else build_grid(
                self.factors,
                method="sobol",
                n_samples=n_candidates,
                seed=seed,
            )
        )
        if not pool:
            msg = "recommend: no candidates to score"
            raise ValueError(msg)
        preds = self.predict_batch(regime, pool)[objective]
        idx = int(np.argmin(preds)) if mode == "min" else int(np.argmax(preds))
        return dict(pool[idx])


def fit_regime_surrogate(  # ruff: ignore[too-many-arguments]
    results: ResultsTable,
    regime_factors: list[Factor],
    factors: list[Factor],
    *,
    method: str = "gp",
    seed: int = 0,
    n_estimators: int = 200,
    cv_folds: int = 5,
    warn_below_r2: float | None = 0.0,
) -> RegimeSurrogate:
    """Fit a surrogate that conditions on regime features.

    Internally fits a single :class:`SurrogateModel` over the joint
    ``regime_factors + factors`` space, so observables can be
    interpolated across continuous regime axes.

    Every config in ``results.configs`` must contain values for both the
    regime features and the design factors.

    Args:
        results: A :class:`ResultsTable` from previous study runs that
            spans multiple regimes.
        regime_factors: Factors describing the regime (additional input
            dimensions of the surrogate; typically continuous).
        factors: Tunable design factors. Together with
            ``regime_factors`` these must cover every key referenced in
            ``results.configs``.
        method: Surrogate backend, ``"gp"`` or ``"rf"``. See
            :func:`trade_study.fit_surrogate`.
        seed: Random seed forwarded to the backend estimators.
        n_estimators: Number of trees for the ``"rf"`` backend.
        cv_folds: Cross-validation folds for the held-out accuracy check
            (#114). See :func:`trade_study.fit_surrogate`.
        warn_below_r2: Warn if any observable's cross-validated R^2 falls
            below this threshold. See :func:`trade_study.fit_surrogate`.

    Returns:
        A fitted :class:`RegimeSurrogate`.

    Raises:
        ValueError: If ``regime_factors`` is empty, if a name appears in
            both ``regime_factors`` and ``factors``, or if the
            underlying :func:`fit_surrogate` call fails.
    """
    if not regime_factors:
        msg = "fit_regime_surrogate: regime_factors must be non-empty"
        raise ValueError(msg)
    overlap = {f.name for f in regime_factors} & {f.name for f in factors}
    if overlap:
        msg = (
            f"fit_regime_surrogate: names appear in both regime_factors and "
            f"factors: {sorted(overlap)}"
        )
        raise ValueError(msg)
    inner = fit_surrogate(
        results,
        [*regime_factors, *factors],
        method=method,
        seed=seed,
        n_estimators=n_estimators,
        cv_folds=cv_folds,
        warn_below_r2=warn_below_r2,
    )
    return RegimeSurrogate(
        inner=inner,
        regime_factors=list(regime_factors),
        factors=list(factors),
    )


def _aggregate_factor_values(factor: Factor, values: list[Any]) -> Any:  # ruff: ignore[any-type]
    """Aggregate one factor's per-regime best values into a bucket value.

    Continuous/discrete (numeric) factors are aggregated by median;
    categorical factors by mode (most common value, ties broken by
    first occurrence).

    Returns:
        The aggregated value for this factor.
    """
    if factor.factor_type == FactorType.CATEGORICAL:
        return Counter(values).most_common(1)[0][0]
    return type(values[0])(np.median(values))


def recommend_bucketed_config(  # ruff: ignore[too-many-arguments]
    regimes: dict[str, dict[str, Any]],
    bucket_fn: Callable[[str, dict[str, Any]], str],
    world_factory: Callable[[dict[str, Any]], Simulator],
    scorer: Scorer,
    factors: list[Factor],
    observables: list[Observable],
    *,
    primary: str,
    n_trials: int = 30,
    n_reps: int = 1,
    seed: int = 42,
) -> dict[str, dict[str, Any]]:
    """Recommend a config per named bucket via per-regime adaptive search (#123).

    The discrete counterpart to :func:`fit_regime_surrogate`: for a
    handful of named regimes too sparse (and too far outside any
    existing training data) for a surrogate to extrapolate across
    sensibly, this instead runs :func:`~trade_study.run_adaptive` (NSGA-II)
    independently per regime, picks each regime's best-found config by
    ``primary``, groups regimes into named buckets via ``bucket_fn``, and
    aggregates each bucket's per-regime best configs (median for
    continuous/discrete factors, mode for categorical) into one
    recommended config per bucket.

    Args:
        regimes: Mapping from regime name to a regime descriptor dict
            (whatever ``world_factory`` needs to fix that regime).
        bucket_fn: Maps ``(regime_name, regime_dict)`` to a bucket name;
            regimes sharing a bucket name have their best configs
            aggregated together.
        world_factory: Builds a regime-scoped :class:`Simulator` from a
            regime dict, e.g. ``lambda r: MySimulator(regime_defaults=r)``.
        scorer: Scorer for observables (shared across all regimes).
        factors: Tunable factors searched by ``run_adaptive`` at each
            regime (the regime itself is fixed via ``world_factory``, not
            part of this search space).
        observables: Observable definitions passed to ``run_adaptive``.
        primary: Name of the observable used to pick each regime's single
            best trial (the one minimizing/maximizing it, per that
            observable's ``direction``) from ``run_adaptive``'s Pareto set.
        n_trials: Optuna trials per regime.
        n_reps: Replicate draws averaged per trial (#122) -- see
            :func:`~trade_study.run_adaptive`'s ``n_reps`` for the full
            rationale. Default 1 (a single draw per trial).
        seed: Random seed forwarded to each regime's ``run_adaptive`` call.

    Returns:
        Mapping from bucket name to its aggregated recommended config.

    Raises:
        ValueError: If ``regimes`` is empty, or ``primary`` doesn't match
            any name in ``observables``.
    """
    if not regimes:
        msg = "recommend_bucketed_config: regimes must be non-empty"
        raise ValueError(msg)
    matching = [o for o in observables if o.name == primary]
    if not matching:
        msg = f"primary={primary!r} not found in observables"
        raise ValueError(msg)
    minimize = matching[0].direction == Direction.MINIMIZE

    per_regime_best: dict[str, dict[str, Any]] = {}
    bucket_members: dict[str, list[str]] = defaultdict(list)
    for name, regime in regimes.items():
        world = world_factory(regime)
        table = run_adaptive(
            world,
            scorer,
            factors,
            observables,
            n_trials=n_trials,
            n_reps=n_reps,
            seed=seed,
        )
        primary_col = table.observable_names.index(primary)
        best_i = (
            int(np.argmin(table.scores[:, primary_col]))
            if minimize
            else int(np.argmax(table.scores[:, primary_col]))
        )
        per_regime_best[name] = table.configs[best_i]
        bucket_members[bucket_fn(name, regime)].append(name)

    factors_by_name = {f.name: f for f in factors}
    return {
        bucket: {
            key: _aggregate_factor_values(
                factors_by_name[key],
                [per_regime_best[member][key] for member in members],
            )
            for key in factors_by_name
        }
        for bucket, members in bucket_members.items()
    }
