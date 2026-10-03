"""Surrogate models over the results table (#82).

Fit a cheap regression model to a :class:`ResultsTable` so observables
can be predicted at untested configurations. Two backends:

- ``"gp"``: scikit-learn :class:`GaussianProcessRegressor` per observable
  with a Matern(1.5) + WhiteKernel; provides predictive standard
  deviations via :meth:`SurrogateModel.uncertainty`.
- ``"rf"``: scikit-learn :class:`RandomForestRegressor` per observable;
  fast, handles non-stationary surfaces, but does not expose calibrated
  uncertainties through :meth:`SurrogateModel.uncertainty`.

Categorical and discrete factors are one-hot encoded; continuous factors
are min-max scaled to ``[0, 1]`` using their declared bounds. The fitted
model is independent per observable column.

Optional dependency: install via the ``trade-study[surrogate]`` extra.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ._checkpoint import _value
from .design import Factor, FactorType, value_to_unit

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from .protocols import ResultsTable


_SUPPORTED_METHODS: frozenset[str] = frozenset({"gp", "rf"})


@dataclass(frozen=True)
class _FactorEncoder:
    """Encodes a list of factors into a numeric design matrix.

    Continuous factors are min-max scaled to ``[0, 1]``. Categorical and
    discrete factors are one-hot encoded with a stable level ordering
    taken from each factor's ``levels`` attribute.

    Attributes:
        factors: Ordered list of factors used for encoding.
        column_names: Flat list of encoded column names (for debugging /
            feature-importance inspection).
    """

    factors: list[Factor]
    column_names: list[str] = field(default_factory=list)

    @classmethod
    def from_factors(cls, factors: list[Factor]) -> _FactorEncoder:
        """Build an encoder with column names derived from ``factors``.

        Args:
            factors: Factor list with bounds (continuous) or levels
                (categorical/discrete) populated.

        Returns:
            A new :class:`_FactorEncoder`.
        """
        cols: list[str] = []
        for f in factors:
            if f.factor_type == FactorType.CONTINUOUS:
                cols.append(f.name)
            else:
                assert f.levels is not None  # ruff: ignore[assert] -- enforced by Factor
                cols.extend(f"{f.name}={lvl!r}" for lvl in f.levels)
        return cls(factors=factors, column_names=cols)

    def transform(
        self,
        configs: Sequence[dict[str, Any]],
    ) -> NDArray[np.float64]:
        """Encode configs as a 2-D numeric design matrix.

        Args:
            configs: Sequence of factor-keyed config dicts.

        Returns:
            Array of shape ``(len(configs), len(column_names))``.

        Raises:
            KeyError: If a config is missing a required factor.
            ValueError: If a categorical/discrete value is not one of the
                factor's declared levels.
        """
        rows: list[list[float]] = []
        for cfg in configs:
            row: list[float] = []
            for f in self.factors:
                if f.name not in cfg:
                    msg = f"config is missing factor {f.name!r}"
                    raise KeyError(msg)
                if f.factor_type == FactorType.CONTINUOUS:
                    row.append(value_to_unit(f, float(cfg[f.name])))
                else:
                    assert f.levels is not None  # ruff: ignore[assert] -- enforced
                    value = cfg[f.name]
                    if value not in f.levels:
                        msg = (
                            f"config value {value!r} for factor {f.name!r} "
                            f"is not in declared levels {f.levels}"
                        )
                        raise ValueError(msg)
                    row.extend(1.0 if value == lvl else 0.0 for lvl in f.levels)
            rows.append(row)
        return np.asarray(rows, dtype=np.float64)


@dataclass(frozen=True)
class PredictionSupport:
    """Marginal training-support diagnostics for one query and observable.

    These checks flag extrapolation; they do not establish joint support or
    provide a calibrated uncertainty guarantee.
    """

    config_index: int
    observable: str
    outside_ranges: dict[str, tuple[float, float]]
    unseen_levels: dict[str, Any]

    @property
    def supported(self) -> bool:
        """Whether every factor lies within its observed marginal support.

        Returns:
            True when no range or level check failed.
        """
        return not self.outside_ranges and not self.unseen_levels


@dataclass
class SurrogateModel:
    """Fitted surrogate over a :class:`ResultsTable`.

    Use :func:`fit_surrogate` to construct one. Per-observable backend
    estimators are stored in ``models``; encoding is shared across them.

    Attributes:
        method: ``"gp"`` or ``"rf"``.
        encoder: Factor encoder used at fit time.
        observable_names: Column names of the predicted observables.
        models: One fitted scikit-learn estimator per observable.
        cv_r2: Per-observable held-out cross-validated R^2 (#114). Missing
            for an observable if too few rows were available to run CV
            (fewer than 2 folds); ``float("nan")`` in that case.
        cv_rmse: Per-observable held-out cross-validated RMSE, companion
            to ``cv_r2`` in the observable's original units.
        row_cv_r2: Shuffled row-validation R², even when grouping is selected.
        row_cv_rmse: Shuffled row-validation RMSE in observable units.
        cv_group_by: Grouping factor names, or None for shuffled row validation.
        validation_groups: Group ids aligned with the original results rows.
        support_ranges: Observed continuous bounds per fitted observable.
        support_levels: Observed levels per fitted observable.
    """

    method: str
    encoder: _FactorEncoder
    observable_names: list[str]
    models: list[Any]
    cv_r2: dict[str, float] = field(default_factory=dict)
    cv_rmse: dict[str, float] = field(default_factory=dict)
    row_cv_r2: dict[str, float] = field(default_factory=dict)
    row_cv_rmse: dict[str, float] = field(default_factory=dict)
    cv_group_by: tuple[str, ...] | None = None
    validation_groups: list[int] = field(default_factory=list)
    support_ranges: dict[str, dict[str, tuple[float, float]]] = field(
        default_factory=dict
    )
    support_levels: dict[str, dict[str, list[Any]]] = field(default_factory=dict)

    def support(self, configs: Sequence[dict[str, Any]]) -> list[PredictionSupport]:
        """Inspect observed support without predicting or emitting warnings.

        Args:
            configs: Factor configurations to inspect.

        Returns:
            One diagnostic per configuration and fitted observable. Support
            uses finite training rows for that observable, in raw factor units.

        """
        reports = []
        for index, config in enumerate(configs):
            for observable in self.observable_names:
                outside = {
                    name: bounds
                    for name, bounds in self.support_ranges.get(observable, {}).items()
                    if not bounds[0] <= float(config[name]) <= bounds[1]
                }
                unseen = {
                    name: config[name]
                    for name, levels in self.support_levels.get(observable, {}).items()
                    if config[name] not in levels
                }
                reports.append(PredictionSupport(index, observable, outside, unseen))
        return reports

    def _encode(
        self, configs: Sequence[dict[str, Any]], *, warn_support: bool
    ) -> NDArray[np.float64]:
        x = self.encoder.transform(configs)
        if warn_support:
            reports = [r for r in self.support(configs) if not r.supported]
            if reports:
                factors = sorted({
                    name
                    for r in reports
                    for name in (*r.outside_ranges, *r.unseen_levels)
                })
                warnings.warn(
                    f"Query outside observed training support for factors {factors}; "
                    "predictions extrapolate and may be unreliable. Inspect support().",
                    UserWarning,
                    stacklevel=3,
                )
        return x

    def predict(
        self, config: dict[str, Any], *, warn_support: bool = True
    ) -> dict[str, float]:
        """Predict observables for a single config.

        Args:
            config: Factor-keyed config dict.
            warn_support: Warn on values outside observed training support.

        Returns:
            Mapping from observable name to predicted scalar.
        """
        x = self._encode([config], warn_support=warn_support)
        return {
            name: float(model.predict(x)[0])
            for name, model in zip(self.observable_names, self.models, strict=True)
        }

    def predict_batch(
        self,
        configs: Sequence[dict[str, Any]],
        *,
        warn_support: bool = True,
    ) -> dict[str, NDArray[np.float64]]:
        """Predict observables for a batch of configs.

        Args:
            configs: Sequence of factor-keyed config dicts.
            warn_support: Warn on values outside observed training support.

        Returns:
            Mapping from observable name to a length-``len(configs)``
            array of predictions.
        """
        x = self._encode(configs, warn_support=warn_support)
        return {
            name: np.asarray(model.predict(x), dtype=np.float64)
            for name, model in zip(self.observable_names, self.models, strict=True)
        }

    def spread_batch(
        self,
        configs: Sequence[dict[str, Any]],
        *,
        warn_support: bool = True,
    ) -> dict[str, NDArray[np.float64]]:
        """Predictive spread per observable for a batch of configs (#115).

        For ``method="gp"`` this is the predictive standard deviation; for
        ``method="rf"`` it is the standard deviation of the individual trees'
        predictions, a relative (not calibrated) measure of disagreement.

        Args:
            configs: Sequence of factor-keyed config dicts.
            warn_support: Warn on values outside observed training support.

        Returns:
            Mapping from observable name to a length-``len(configs)`` array.
        """
        x = self._encode(configs, warn_support=warn_support)
        out: dict[str, NDArray[np.float64]] = {}
        for name, model in zip(self.observable_names, self.models, strict=True):
            if self.method == "gp":
                _, std = model.predict(x, return_std=True)
            else:
                std = np.std([tree.predict(x) for tree in model.estimators_], axis=0)
            out[name] = np.asarray(std, dtype=np.float64)
        return out

    def uncertainty(
        self, config: dict[str, Any], *, warn_support: bool = True
    ) -> dict[str, float]:
        """Predictive standard deviation per observable (GP only).

        Args:
            config: Factor-keyed config dict.
            warn_support: Warn on values outside observed training support.

        Returns:
            Mapping from observable name to predictive standard deviation.

        Raises:
            NotImplementedError: If the backend does not expose calibrated
                uncertainties (currently anything other than ``"gp"``).
        """
        if self.method != "gp":
            msg = (
                f"uncertainty() is only supported for method='gp'; "
                f"this surrogate uses method={self.method!r}"
            )
            raise NotImplementedError(msg)
        x = self._encode([config], warn_support=warn_support)
        out: dict[str, float] = {}
        for name, model in zip(self.observable_names, self.models, strict=True):
            _, std = model.predict(x, return_std=True)
            out[name] = float(std[0])
        return out


def fit_surrogate(  # ruff: ignore[too-many-arguments]
    results: ResultsTable,
    factors: list[Factor],
    *,
    method: str = "gp",
    seed: int = 0,
    n_estimators: int = 200,
    cv_folds: int = 5,
    warn_below_r2: float | None = 0.0,
    cv_group_by: str | Sequence[str] | None = None,
) -> SurrogateModel:
    """Fit a per-observable surrogate over a :class:`ResultsTable`.

    Rows whose score column contains ``NaN`` are dropped on a
    per-observable basis (so a partially-evaluated trial still
    contributes to the observables it does have).

    After fitting each observable's model on all its available rows,
    also computes a held-out cross-validated R^2/RMSE (#114) so callers
    can tell a well-fit surrogate from one that's effectively guessing
    in a sparse or noisy region of the design space -- neither backend
    reports this on its own (RF's cheap OOB score isn't available for
    GP, so CV is used uniformly for both, at the cost of ``cv_folds``
    extra fits per observable).

    Args:
        results: A :class:`ResultsTable` from a previous study run.
        factors: Factor definitions used to encode ``results.configs``.
            Must cover every key in the configs.
        method: ``"gp"`` for Gaussian process (Matern 1.5 + WhiteKernel)
            or ``"rf"`` for a random forest.
        seed: Random seed forwarded to the backend estimators.
        n_estimators: Number of trees for the ``"rf"`` backend; ignored
            for ``"gp"``.
        cv_folds: Number of cross-validation folds used to compute
            ``cv_r2``/``cv_rmse``. Clamped down to the observable's
            available row count when fewer than ``cv_folds`` rows exist;
            skipped (``nan``) for an observable with fewer than 2 rows.
        warn_below_r2: If not ``None``, emit a ``UserWarning`` naming any
            observable whose ``cv_r2`` falls below this threshold --
            0.0 (the default) flags a surrogate that predicts no better
            than the training mean. Pass ``None`` to disable.
        cv_group_by: ``"design"`` groups identical factor configurations;
            a sequence of factor names groups by those regime descriptors.
            Grouped folds keep all rows of a group together. When selected,
            ``cv_r2``/``cv_rmse`` report grouped validation; ``row_cv_r2`` and
            ``row_cv_rmse`` retain shuffled row validation for comparison.
            ``None`` preserves the existing row-validation behavior.

    Returns:
        A fitted :class:`SurrogateModel`.

    Raises:
        ValueError: If ``method`` is unknown, ``results`` is empty, or
            no observable has at least two non-NaN training rows.
    """
    if method not in _SUPPORTED_METHODS:
        msg = (
            f"Unknown surrogate method {method!r}. "
            f"Supported: {sorted(_SUPPORTED_METHODS)}"
        )
        raise ValueError(msg)
    if not results.configs:
        msg = "fit_surrogate: results table is empty"
        raise ValueError(msg)

    encoder = _FactorEncoder.from_factors(factors)
    x_full = encoder.transform(results.configs)
    group_names, groups = _validation_groups(results.configs, factors, cv_group_by)

    surrogate = SurrogateModel(
        method=method,
        encoder=encoder,
        observable_names=[],
        models=[],
        cv_group_by=group_names,
        validation_groups=[] if groups is None else groups.tolist(),
    )
    low_accuracy: list[str] = []
    for j, name in enumerate(results.observable_names):
        y = results.scores[:, j]
        mask = np.isfinite(y)
        n_rows = int(mask.sum())
        if n_rows < 2:
            continue
        model = _make_estimator(method, seed=seed, n_estimators=n_estimators)
        model.fit(x_full[mask], y[mask])
        surrogate.models.append(model)
        surrogate.observable_names.append(name)

        r2, rmse = _cross_val_accuracy(
            x_full[mask],
            y[mask],
            method=method,
            seed=seed,
            n_estimators=n_estimators,
            n_folds=min(cv_folds, n_rows),
        )
        surrogate.row_cv_r2[name], surrogate.row_cv_rmse[name] = r2, rmse
        if groups is not None:
            r2, rmse = _cross_val_accuracy(
                x_full[mask],
                y[mask],
                method=method,
                seed=seed,
                n_estimators=n_estimators,
                n_folds=cv_folds,
                groups=groups[mask],
            )
        configs = [cfg for cfg, keep in zip(results.configs, mask, strict=True) if keep]
        surrogate.support_ranges[name], surrogate.support_levels[name] = (
            _training_support(configs, factors)
        )
        surrogate.cv_r2[name] = r2
        surrogate.cv_rmse[name] = rmse
        if warn_below_r2 is not None and np.isfinite(r2) and r2 < warn_below_r2:
            low_accuracy.append(name)

    if not surrogate.models:
        msg = (
            "fit_surrogate: no observable has at least 2 non-NaN training "
            "rows; nothing to fit"
        )
        raise ValueError(msg)

    if low_accuracy:
        warnings.warn(
            f"fit_surrogate: cross-validated R^2 below {warn_below_r2} for "
            f"{low_accuracy} -- predictions for these observables may not "
            f"be trustworthy. See SurrogateModel.cv_r2 for exact values.",
            UserWarning,
            stacklevel=2,
        )

    return surrogate


def _validation_groups(
    configs: list[dict[str, Any]],
    factors: list[Factor],
    group_by: str | Sequence[str] | None,
) -> tuple[tuple[str, ...] | None, NDArray[np.int64] | None]:
    """Create stable group ids from selected factor values.

    Returns:
        Grouping factor names and an aligned integer group vector, or None.

    Raises:
        ValueError: If grouping names are empty, duplicated, or unknown.
    """
    if group_by is None:
        return None, None
    names = tuple(f.name for f in factors) if group_by == "design" else tuple(group_by)
    if isinstance(group_by, str) and group_by != "design":
        msg = "cv_group_by must be 'design' or a sequence of factor names"
        raise ValueError(msg)
    if (
        not names
        or len(set(names)) != len(names)
        or set(names) - {f.name for f in factors}
    ):
        msg = "cv_group_by must contain distinct known factor names"
        raise ValueError(msg)
    identities: dict[str, int] = {}
    definitions = {f.name: f for f in factors}
    groups = []
    for config in configs:
        key = json.dumps(
            [_group_value(definitions[name], config) for name in names],
            sort_keys=True,
            allow_nan=False,
        )
        groups.append(identities.setdefault(key, len(identities)))
    return names, np.asarray(groups, dtype=np.int64)


def _group_value(factor: Factor, config: dict[str, Any]) -> object:
    if factor.factor_type == FactorType.CONTINUOUS:
        return float(config[factor.name])
    return _value(
        next(level for level in (factor.levels or []) if level == config[factor.name]),
        None,
    )


def _training_support(
    configs: list[dict[str, Any]], factors: list[Factor]
) -> tuple[dict[str, tuple[float, float]], dict[str, list[Any]]]:
    ranges = {}
    levels = {}
    for factor in factors:
        values = [config[factor.name] for config in configs]
        if factor.factor_type == FactorType.CONTINUOUS:
            ranges[factor.name] = min(map(float, values)), max(map(float, values))
        else:
            levels[factor.name] = [
                level for level in (factor.levels or []) if level in values
            ]
    return ranges, levels


def _cross_val_accuracy(
    x: NDArray[np.floating[Any]],
    y: NDArray[np.floating[Any]],
    *,
    method: str,
    seed: int,
    n_estimators: int,
    n_folds: int,
    groups: NDArray[np.int64] | None = None,
) -> tuple[float, float]:
    """K-fold cross-validated R^2/RMSE for one observable.

    Returns:
        ``(r2, rmse)``, both ``float("nan")`` if fewer than 2 folds are
        possible.
    """
    n_folds = min(n_folds, len(y) if groups is None else len(np.unique(groups)))
    if n_folds < 2:
        return float("nan"), float("nan")

    from sklearn.model_selection import GroupKFold, KFold  # type: ignore[import-untyped]

    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    splits = (
        kfold.split(x)
        if groups is None
        else GroupKFold(n_splits=n_folds).split(x, y, groups)
    )
    y_pred = np.empty(y.shape, dtype=np.float64)
    for train_idx, test_idx in splits:
        fold_model = _make_estimator(method, seed=seed, n_estimators=n_estimators)
        fold_model.fit(x[train_idx], y[train_idx])
        y_pred[test_idx] = fold_model.predict(x[test_idx])

    residuals = y - y_pred
    rmse = float(np.sqrt(np.mean(residuals**2)))
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return r2, rmse


def _make_estimator(method: str, *, seed: int, n_estimators: int) -> Any:  # ruff: ignore[any-type]
    """Construct an unfitted scikit-learn estimator for the requested method.

    Args:
        method: ``"gp"`` or ``"rf"`` (validated by the caller).
        seed: Random seed.
        n_estimators: Number of trees (RF only).

    Returns:
        An unfitted scikit-learn regressor.
    """
    if method == "gp":
        from sklearn.gaussian_process import (  # type: ignore[import-untyped]
            GaussianProcessRegressor,
        )
        from sklearn.gaussian_process.kernels import (  # type: ignore[import-untyped]
            ConstantKernel,
            Matern,
            WhiteKernel,
        )

        kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=1.5) + WhiteKernel(
            noise_level=1e-3,
        )
        return GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            random_state=seed,
            n_restarts_optimizer=2,
        )
    from sklearn.ensemble import (  # type: ignore[import-untyped]
        RandomForestRegressor,
    )

    return RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=seed,
    )
