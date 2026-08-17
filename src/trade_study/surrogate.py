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

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from .design import Factor, FactorType

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
                    assert f.bounds is not None  # ruff: ignore[assert] -- enforced
                    lo, hi = f.bounds
                    row.append((float(cfg[f.name]) - lo) / (hi - lo))
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
    """

    method: str
    encoder: _FactorEncoder
    observable_names: list[str]
    models: list[Any]

    def predict(self, config: dict[str, Any]) -> dict[str, float]:
        """Predict observables for a single config.

        Args:
            config: Factor-keyed config dict.

        Returns:
            Mapping from observable name to predicted scalar.
        """
        x = self.encoder.transform([config])
        return {
            name: float(model.predict(x)[0])
            for name, model in zip(self.observable_names, self.models, strict=True)
        }

    def predict_batch(
        self,
        configs: Sequence[dict[str, Any]],
    ) -> dict[str, NDArray[np.float64]]:
        """Predict observables for a batch of configs.

        Args:
            configs: Sequence of factor-keyed config dicts.

        Returns:
            Mapping from observable name to a length-``len(configs)``
            array of predictions.
        """
        x = self.encoder.transform(configs)
        return {
            name: np.asarray(model.predict(x), dtype=np.float64)
            for name, model in zip(self.observable_names, self.models, strict=True)
        }

    def uncertainty(self, config: dict[str, Any]) -> dict[str, float]:
        """Predictive standard deviation per observable (GP only).

        Args:
            config: Factor-keyed config dict.

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
        x = self.encoder.transform([config])
        out: dict[str, float] = {}
        for name, model in zip(self.observable_names, self.models, strict=True):
            _, std = model.predict(x, return_std=True)
            out[name] = float(std[0])
        return out


def fit_surrogate(
    results: ResultsTable,
    factors: list[Factor],
    *,
    method: str = "gp",
    seed: int = 0,
    n_estimators: int = 200,
) -> SurrogateModel:
    """Fit a per-observable surrogate over a :class:`ResultsTable`.

    Rows whose score column contains ``NaN`` are dropped on a
    per-observable basis (so a partially-evaluated trial still
    contributes to the observables it does have).

    Args:
        results: A :class:`ResultsTable` from a previous study run.
        factors: Factor definitions used to encode ``results.configs``.
            Must cover every key in the configs.
        method: ``"gp"`` for Gaussian process (Matern 1.5 + WhiteKernel)
            or ``"rf"`` for a random forest.
        seed: Random seed forwarded to the backend estimators.
        n_estimators: Number of trees for the ``"rf"`` backend; ignored
            for ``"gp"``.

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

    models: list[Any] = []
    fitted_obs: list[str] = []
    for j, name in enumerate(results.observable_names):
        y = results.scores[:, j]
        mask = ~np.isnan(y)
        if int(mask.sum()) < 2:
            continue
        model = _make_estimator(method, seed=seed, n_estimators=n_estimators)
        model.fit(x_full[mask], y[mask])
        models.append(model)
        fitted_obs.append(name)

    if not models:
        msg = (
            "fit_surrogate: no observable has at least 2 non-NaN training "
            "rows; nothing to fit"
        )
        raise ValueError(msg)

    return SurrogateModel(
        method=method,
        encoder=encoder,
        observable_names=fitted_obs,
        models=models,
    )


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
