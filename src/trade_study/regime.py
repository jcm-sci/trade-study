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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .design import Factor, build_grid
from .surrogate import SurrogateModel, fit_surrogate

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from .protocols import ResultsTable


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


def fit_regime_surrogate(
    results: ResultsTable,
    regime_factors: list[Factor],
    factors: list[Factor],
    *,
    method: str = "gp",
    seed: int = 0,
    n_estimators: int = 200,
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
    )
    return RegimeSurrogate(
        inner=inner,
        regime_factors=list(regime_factors),
        factors=list(factors),
    )
