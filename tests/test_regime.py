"""Tests for regime-conditional surrogate (#105)."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pytest

from trade_study import (
    Direction,
    Factor,
    FactorType,
    Observable,
    RegimeSurrogate,
    build_grid,
    fit_regime_surrogate,
    recommend_bucketed_config,
    run_grid,
)

if TYPE_CHECKING:
    from trade_study.protocols import ResultsTable


class _PassWorld:
    """Trivial simulator: passes config through."""

    def generate(
        self,
        config: dict[str, object],
    ) -> tuple[dict[str, object], dict[str, object]]:
        return config, config


class _RegimeScorer:
    """Scorer where ``loss = (lr - 0.1*n)**2``.

    The optimal ``lr`` is a linear function of regime feature ``n``.
    """

    def score(
        self,
        truth: object,
        observations: dict[str, object],
        config: dict[str, object],
    ) -> dict[str, float]:
        del truth, config
        n = float(observations["n"])
        lr = float(observations["lr"])
        return {"loss": (lr - 0.1 * n) ** 2}


@pytest.fixture
def regime_factor() -> Factor:
    return Factor("n", FactorType.CONTINUOUS, bounds=(0.0, 10.0))


@pytest.fixture
def design_factor() -> Factor:
    return Factor("lr", FactorType.CONTINUOUS, bounds=(0.0, 1.0))


def _make_results(
    regime_factor: Factor,
    design_factor: Factor,
    n: int = 64,
    seed: int = 0,
) -> ResultsTable:
    grid = build_grid(
        [regime_factor, design_factor],
        method="sobol",
        n_samples=n,
        seed=seed,
    )
    obs = [Observable("loss", Direction.MINIMIZE)]
    return run_grid(_PassWorld(), _RegimeScorer(), grid, obs)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_fit_requires_regime_factors(
    regime_factor: Factor,
    design_factor: Factor,
) -> None:
    results = _make_results(regime_factor, design_factor, n=8)
    with pytest.raises(ValueError, match="regime_factors must be non-empty"):
        fit_regime_surrogate(results, [], [design_factor])


def test_fit_rejects_overlapping_names(
    regime_factor: Factor,
    design_factor: Factor,
) -> None:
    results = _make_results(regime_factor, design_factor, n=8)
    dup = Factor("n", FactorType.CONTINUOUS, bounds=(0.0, 10.0))
    with pytest.raises(ValueError, match="appear in both"):
        fit_regime_surrogate(results, [regime_factor], [dup])


# ---------------------------------------------------------------------------
# Cross-validated accuracy passthrough (#114)
# ---------------------------------------------------------------------------


def test_cv_r2_rmse_passthrough(
    regime_factor: Factor,
    design_factor: Factor,
) -> None:
    results = _make_results(regime_factor, design_factor, n=64)
    sur = fit_regime_surrogate(
        results,
        [regime_factor],
        [design_factor],
        method="rf",
        seed=0,
    )
    assert sur.cv_r2 == sur.inner.cv_r2
    assert sur.cv_rmse == sur.inner.cv_rmse
    assert "loss" in sur.cv_r2
    assert "loss" in sur.cv_rmse


def test_recommend_warns_on_poor_fit(
    regime_factor: Factor,
    design_factor: Factor,
) -> None:
    results = _make_results(regime_factor, design_factor, n=16)
    rng = np.random.default_rng(0)
    results.scores[:, 0] = rng.standard_normal(len(results.configs))
    sur = fit_regime_surrogate(
        results,
        [regime_factor],
        [design_factor],
        method="rf",
        seed=0,
        warn_below_r2=None,  # suppress the fit-time warning to isolate recommend()'s
    )
    with pytest.warns(UserWarning, match="cross-validated R\\^2"):
        sur.recommend({"n": 5.0}, objective="loss")


def test_recommend_warn_below_r2_none_disables(
    regime_factor: Factor,
    design_factor: Factor,
) -> None:
    results = _make_results(regime_factor, design_factor, n=16)
    rng = np.random.default_rng(0)
    results.scores[:, 0] = rng.standard_normal(len(results.configs))
    sur = fit_regime_surrogate(
        results,
        [regime_factor],
        [design_factor],
        method="rf",
        seed=0,
        warn_below_r2=None,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sur.recommend({"n": 5.0}, objective="loss", warn_below_r2=None)


# ---------------------------------------------------------------------------
# Predict / uncertainty
# ---------------------------------------------------------------------------


def test_predict_returns_observable_dict(
    regime_factor: Factor,
    design_factor: Factor,
) -> None:
    results = _make_results(regime_factor, design_factor, n=32)
    sur = fit_regime_surrogate(
        results,
        [regime_factor],
        [design_factor],
        method="rf",
        seed=0,
    )
    assert isinstance(sur, RegimeSurrogate)
    pred = sur.predict({"n": 5.0}, {"lr": 0.5})
    assert set(pred) == {"loss"}
    assert isinstance(pred["loss"], float)


def test_predict_batch_shape(
    regime_factor: Factor,
    design_factor: Factor,
) -> None:
    results = _make_results(regime_factor, design_factor, n=32)
    sur = fit_regime_surrogate(
        results,
        [regime_factor],
        [design_factor],
        method="rf",
        seed=0,
    )
    pred = sur.predict_batch({"n": 5.0}, [{"lr": x} for x in [0.0, 0.25, 0.75]])
    assert pred["loss"].shape == (3,)


def test_gp_uncertainty_returns_floats(
    regime_factor: Factor,
    design_factor: Factor,
) -> None:
    results = _make_results(regime_factor, design_factor, n=32)
    sur = fit_regime_surrogate(
        results,
        [regime_factor],
        [design_factor],
        method="gp",
        seed=0,
    )
    unc = sur.uncertainty({"n": 5.0}, {"lr": 0.5})
    assert set(unc) == {"loss"}
    assert unc["loss"] >= 0.0


def test_rf_uncertainty_raises(
    regime_factor: Factor,
    design_factor: Factor,
) -> None:
    results = _make_results(regime_factor, design_factor, n=16)
    sur = fit_regime_surrogate(
        results,
        [regime_factor],
        [design_factor],
        method="rf",
        seed=0,
    )
    with pytest.raises(NotImplementedError):
        sur.uncertainty({"n": 5.0}, {"lr": 0.5})


# ---------------------------------------------------------------------------
# Recommend
# ---------------------------------------------------------------------------


def test_recommend_tracks_regime(
    regime_factor: Factor,
    design_factor: Factor,
) -> None:
    """At ``n=2`` the optimum is ``lr~=0.2``; at ``n=8`` it is ``lr~=0.8``."""
    results = _make_results(regime_factor, design_factor, n=128, seed=0)
    sur = fit_regime_surrogate(
        results,
        [regime_factor],
        [design_factor],
        method="rf",
        seed=0,
        n_estimators=200,
    )
    low = sur.recommend({"n": 2.0}, objective="loss", n_candidates=128, seed=1)
    high = sur.recommend({"n": 8.0}, objective="loss", n_candidates=128, seed=1)
    assert low["lr"] < high["lr"]
    assert low["lr"] == pytest.approx(0.2, abs=0.2)
    assert high["lr"] == pytest.approx(0.8, abs=0.2)


def test_recommend_mode_max_inverts_choice(
    regime_factor: Factor,
    design_factor: Factor,
) -> None:
    results = _make_results(regime_factor, design_factor, n=64, seed=0)
    sur = fit_regime_surrogate(
        results,
        [regime_factor],
        [design_factor],
        method="rf",
        seed=0,
    )
    pool = [{"lr": 0.1}, {"lr": 0.5}, {"lr": 0.9}]
    best_min = sur.recommend({"n": 5.0}, objective="loss", mode="min", candidates=pool)
    best_max = sur.recommend({"n": 5.0}, objective="loss", mode="max", candidates=pool)
    preds = sur.predict_batch({"n": 5.0}, pool)["loss"]
    assert best_min["lr"] == pool[int(np.argmin(preds))]["lr"]
    assert best_max["lr"] == pool[int(np.argmax(preds))]["lr"]


def test_recommend_rejects_unknown_objective(
    regime_factor: Factor,
    design_factor: Factor,
) -> None:
    results = _make_results(regime_factor, design_factor, n=16)
    sur = fit_regime_surrogate(
        results,
        [regime_factor],
        [design_factor],
        method="rf",
        seed=0,
    )
    with pytest.raises(ValueError, match="not a fitted observable"):
        sur.recommend({"n": 5.0}, objective="bogus")


def test_recommend_rejects_bad_mode(
    regime_factor: Factor,
    design_factor: Factor,
) -> None:
    results = _make_results(regime_factor, design_factor, n=16)
    sur = fit_regime_surrogate(
        results,
        [regime_factor],
        [design_factor],
        method="rf",
        seed=0,
    )
    with pytest.raises(ValueError, match="mode must be"):
        sur.recommend({"n": 5.0}, objective="loss", mode="bogus")


def test_recommend_rejects_empty_candidates(
    regime_factor: Factor,
    design_factor: Factor,
) -> None:
    results = _make_results(regime_factor, design_factor, n=16)
    sur = fit_regime_surrogate(
        results,
        [regime_factor],
        [design_factor],
        method="rf",
        seed=0,
    )
    with pytest.raises(ValueError, match="no candidates"):
        sur.recommend({"n": 5.0}, objective="loss", candidates=[])


# ---------------------------------------------------------------------------
# recommend_bucketed_config (#123)
# ---------------------------------------------------------------------------


class _TargetWorld:
    """Simulator fixed to one regime's target alpha via regime_defaults."""

    def __init__(self, *, target: float, method: str) -> None:
        self._target = target
        self._method = method

    def generate(
        self, config: dict[str, object]
    ) -> tuple[dict[str, object], dict[str, object]]:
        merged = {**config, "target": self._target, "true_method": self._method}
        return merged, merged


class _TargetScorer:
    """cost = (alpha - target)**2; bonus reward when method matches target's."""

    def score(
        self,
        truth: object,
        observations: dict[str, object],
        config: dict[str, object],
    ) -> dict[str, float]:
        del truth, config
        alpha = float(observations["alpha"])
        target = float(observations["target"])
        method_ok = observations["method"] == observations["true_method"]
        return {
            "cost": (alpha - target) ** 2,
            "reward": 1.0 if method_ok else 0.0,
        }


@pytest.fixture
def bucketed_factors() -> list[Factor]:
    """One continuous and one categorical tunable factor.

    Returns:
        List with ``alpha`` (continuous) and ``method`` (categorical).
    """
    return [
        Factor("alpha", FactorType.CONTINUOUS, bounds=(0.0, 1.0)),
        Factor("method", FactorType.CATEGORICAL, levels=["a", "b"]),
    ]


@pytest.fixture
def bucketed_observables() -> list[Observable]:
    """Cost (minimize) and reward (maximize) observables.

    Returns:
        List of two Observable instances.
    """
    return [
        Observable("cost", Direction.MINIMIZE),
        Observable("reward", Direction.MAXIMIZE),
    ]


def test_recommend_bucketed_config_aggregates_median(
    bucketed_factors: list[Factor],
    bucketed_observables: list[Observable],
) -> None:
    regimes = {
        "r1": {"target": 0.2, "method": "a"},
        "r2": {"target": 0.8, "method": "a"},
    }
    result = recommend_bucketed_config(
        regimes,
        bucket_fn=lambda _name, _r: "only",
        world_factory=lambda r: _TargetWorld(target=r["target"], method=r["method"]),
        scorer=_TargetScorer(),
        factors=bucketed_factors,
        observables=bucketed_observables,
        primary="cost",
        n_trials=40,
        seed=0,
    )
    assert set(result.keys()) == {"only"}
    # each regime's best alpha should land near its own target; median of
    # two well-optimized targets (0.2, 0.8) should land near their midpoint.
    assert result["only"]["alpha"] == pytest.approx(0.5, abs=0.2)


def test_recommend_bucketed_config_separate_buckets_stay_separate(
    bucketed_factors: list[Factor],
    bucketed_observables: list[Observable],
) -> None:
    regimes = {
        "r1": {"target": 0.1, "method": "a"},
        "r2": {"target": 0.9, "method": "a"},
    }
    result = recommend_bucketed_config(
        regimes,
        bucket_fn=lambda name, _r: name,  # each regime is its own bucket
        world_factory=lambda r: _TargetWorld(target=r["target"], method=r["method"]),
        scorer=_TargetScorer(),
        factors=bucketed_factors,
        observables=bucketed_observables,
        primary="cost",
        n_trials=40,
        seed=0,
    )
    assert set(result.keys()) == {"r1", "r2"}
    assert result["r1"]["alpha"] == pytest.approx(0.1, abs=0.2)
    assert result["r2"]["alpha"] == pytest.approx(0.9, abs=0.2)


def test_recommend_bucketed_config_categorical_uses_mode(
    bucketed_factors: list[Factor],
    bucketed_observables: list[Observable],
) -> None:
    regimes = {
        "r1": {"target": 0.5, "method": "a"},
        "r2": {"target": 0.5, "method": "a"},
        "r3": {"target": 0.5, "method": "b"},
    }
    result = recommend_bucketed_config(
        regimes,
        bucket_fn=lambda _name, _r: "only",
        world_factory=lambda r: _TargetWorld(target=r["target"], method=r["method"]),
        scorer=_TargetScorer(),
        factors=bucketed_factors,
        observables=bucketed_observables,
        primary="reward",
        n_trials=30,
        seed=0,
    )
    # two of three regimes reward method="a"; mode should pick it.
    assert result["only"]["method"] == "a"


def test_recommend_bucketed_config_respects_maximize_direction(
    bucketed_factors: list[Factor],
    bucketed_observables: list[Observable],
) -> None:
    regimes = {"r1": {"target": 0.5, "method": "a"}}
    result = recommend_bucketed_config(
        regimes,
        bucket_fn=lambda _name, _r: "only",
        world_factory=lambda r: _TargetWorld(target=r["target"], method=r["method"]),
        scorer=_TargetScorer(),
        factors=bucketed_factors,
        observables=bucketed_observables,
        primary="reward",
        n_trials=20,
        seed=0,
    )
    assert result["only"]["method"] == "a"


def test_recommend_bucketed_config_rejects_empty_regimes(
    bucketed_factors: list[Factor],
    bucketed_observables: list[Observable],
) -> None:
    with pytest.raises(ValueError, match="regimes must be non-empty"):
        recommend_bucketed_config(
            {},
            bucket_fn=lambda _n, _r: "b",
            world_factory=lambda r: _TargetWorld(
                target=r["target"], method=r["method"]
            ),
            scorer=_TargetScorer(),
            factors=bucketed_factors,
            observables=bucketed_observables,
            primary="cost",
        )


def test_recommend_bucketed_config_rejects_unknown_primary(
    bucketed_factors: list[Factor],
    bucketed_observables: list[Observable],
) -> None:
    regimes = {"r1": {"target": 0.5, "method": "a"}}
    with pytest.raises(ValueError, match="not found in observables"):
        recommend_bucketed_config(
            regimes,
            bucket_fn=lambda _n, _r: "b",
            world_factory=lambda r: _TargetWorld(
                target=r["target"], method=r["method"]
            ),
            scorer=_TargetScorer(),
            factors=bucketed_factors,
            observables=bucketed_observables,
            primary="bogus",
        )
