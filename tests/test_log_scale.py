"""Tests for log-scale continuous factors (#131)."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import optuna
import pytest
from scipy import stats

from trade_study.design import (
    Factor,
    FactorType,
    build_grid,
    screen,
    unit_to_value,
    value_to_unit,
)
from trade_study.protocols import Direction, Observable
from trade_study.runner import run_adaptive

_surrogate = importlib.import_module("trade_study.surrogate")

LOW, HIGH = 1e-3, 10.0


def _log_factor(name: str = "prior") -> Factor:
    return Factor(name, FactorType.CONTINUOUS, bounds=(LOW, HIGH), log_scale=True)


def test_log_scale_requires_positive_bounds() -> None:
    with pytest.raises(ValueError, match="positive bounds"):
        Factor("x", FactorType.CONTINUOUS, bounds=(0.0, 1.0), log_scale=True)


def test_log_scale_requires_a_continuous_factor() -> None:
    with pytest.raises(ValueError, match="log_scale applies to continuous"):
        Factor("x", FactorType.DISCRETE, levels=[1, 10], log_scale=True)


def test_unit_mapping_is_geometric_and_invertible() -> None:
    factor = _log_factor()

    assert unit_to_value(factor, 0.0) == pytest.approx(LOW)
    assert unit_to_value(factor, 1.0) == pytest.approx(HIGH)
    assert unit_to_value(factor, 0.5) == pytest.approx(np.sqrt(LOW * HIGH))
    for unit in (0.1, 0.37, 0.9):
        assert value_to_unit(factor, unit_to_value(factor, unit)) == pytest.approx(unit)


@pytest.mark.parametrize("method", ["sobol", "halton", "lhs"])
def test_designs_sample_log_uniformly(method: str) -> None:
    factors = [_log_factor(), Factor("rate", FactorType.CONTINUOUS, bounds=(0, 1))]

    grid = build_grid(factors, method=method, n_samples=256, seed=3)
    values = np.array([cfg["prior"] for cfg in grid])

    assert np.all((values >= LOW) & (values <= HIGH))
    uniform = stats.uniform(loc=np.log(LOW), scale=np.log(HIGH) - np.log(LOW))
    assert stats.kstest(np.log(values), uniform.cdf).pvalue > 0.01


def test_linear_factors_are_unchanged() -> None:
    factor = Factor("rate", FactorType.CONTINUOUS, bounds=(2.0, 4.0))

    assert unit_to_value(factor, 0.25) == pytest.approx(2.5)


def test_run_adaptive_requests_log_sampling(monkeypatch: pytest.MonkeyPatch) -> None:
    requested: list[bool] = []
    original = optuna.trial.Trial.suggest_float

    def recording(self: Any, name: str, low: float, high: float, **kwargs: Any) -> Any:
        requested.append(bool(kwargs.get("log")))
        return original(self, name, low, high, **kwargs)

    monkeypatch.setattr(optuna.trial.Trial, "suggest_float", recording)

    class World:
        def generate(self, config: dict[str, Any]) -> tuple[Any, Any]:
            return config, config

    class Scorer:
        def score(self, truth: Any, obs: Any, config: dict[str, Any]) -> dict:
            return {"loss": abs(np.log10(config["prior"]))}

    result = run_adaptive(
        World(),
        Scorer(),
        [_log_factor()],
        [Observable("loss", Direction.MINIMIZE)],
        n_trials=8,
    )

    assert requested
    assert all(requested)
    assert all(LOW <= cfg["prior"] <= HIGH for cfg in result.configs)


def test_screen_samples_log_factors_across_decades() -> None:
    seen: list[float] = []

    def run_fn(cfg: dict[str, Any]) -> dict[str, float]:
        seen.append(cfg["prior"])
        return {"loss": float(np.log10(cfg["prior"]))}

    importance = screen(run_fn, [_log_factor()], method="morris", n_trajectories=20)

    assert min(seen) < 1e-2
    assert max(seen) > 1.0
    assert importance["loss"].shape == (1,)


def test_surrogate_encodes_log_factors_in_log_space() -> None:
    encoder = _surrogate._FactorEncoder.from_factors([_log_factor()])  # ruff: ignore[private-member-access]

    encoded = encoder.transform([{"prior": np.sqrt(LOW * HIGH)}, {"prior": HIGH}])

    np.testing.assert_allclose(encoded[:, 0], [0.5, 1.0])
