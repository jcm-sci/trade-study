"""Experimental design and factor screening.

Wraps pyDOE3 for grid construction and SALib for sensitivity screening.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import product
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


class FactorType(Enum):
    """Type of design factor."""

    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    CATEGORICAL = "categorical"


@dataclass(frozen=True)
class Factor:
    """A single design factor.

    Attributes:
        name: Factor identifier (e.g. "alpha", "layer1_method").
        factor_type: Continuous, discrete, or categorical.
        levels: For categorical/discrete: list of allowed values.
        bounds: For continuous: (low, high) tuple.
    """

    name: str
    factor_type: FactorType
    levels: list[Any] | None = None
    bounds: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        """Validate factor constraints.

        Raises:
            ValueError: If continuous factor missing bounds.
        """
        if self.factor_type == FactorType.CONTINUOUS and self.bounds is None:
            msg = f"Continuous factor '{self.name}' requires bounds"
            raise ValueError(msg)
        if self.factor_type != FactorType.CONTINUOUS and self.levels is None:
            msg = f"Factor '{self.name}' of type {self.factor_type} requires levels"
            raise ValueError(msg)


def build_grid(
    factors: list[Factor],
    *,
    method: str = "full",
    n_samples: int = 100,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Build an experimental design grid.

    Args:
        factors: List of design factors.
        method: Design method. One of:
            - "full": Full factorial (categorical/discrete only).
            - "lhs": Latin hypercube sampling (continuous factors, maps
              categorical factors to uniform random selection).
            - "fractional": Fractional factorial via pyDOE3.
        n_samples: Number of samples for LHS.
        seed: Random seed.

    Returns:
        List of config dictionaries, one per design point.

    Raises:
        ValueError: If an unknown design method is specified.
    """
    if method == "full":
        return _full_factorial(factors)
    if method == "lhs":
        return _latin_hypercube(factors, n_samples=n_samples, seed=seed)
    msg = f"Unknown design method: {method!r}"
    raise ValueError(msg)


def _full_factorial(factors: list[Factor]) -> list[dict[str, Any]]:
    """Full factorial over all factor levels.

    Returns:
        List of config dictionaries, one per design point.

    Raises:
        ValueError: If a factor has bounds instead of levels.
    """
    level_lists = []
    for f in factors:
        if f.levels is not None:
            level_lists.append(f.levels)
        elif f.bounds is not None:
            msg = f"Full factorial requires levels, not bounds, for factor '{f.name}'"
            raise ValueError(msg)
    names = [f.name for f in factors]
    return [dict(zip(names, combo, strict=True)) for combo in product(*level_lists)]


def _latin_hypercube(
    factors: list[Factor],
    *,
    n_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Latin hypercube design via pyDOE3.

    Returns:
        List of config dictionaries, one per design point.
    """
    from pyDOE3 import lhs  # type: ignore[import-untyped]

    rng = np.random.default_rng(seed)
    n_factors = len(factors)
    raw = lhs(n_factors, samples=n_samples, criterion="maximin", random_state=rng)

    configs: list[dict[str, Any]] = []
    for row in raw:
        cfg: dict[str, Any] = {}
        for j, f in enumerate(factors):
            if f.factor_type == FactorType.CONTINUOUS and f.bounds is not None:
                lo, hi = f.bounds
                cfg[f.name] = lo + row[j] * (hi - lo)
            elif f.levels is not None:
                idx = int(row[j] * len(f.levels))
                idx = min(idx, len(f.levels) - 1)
                cfg[f.name] = f.levels[idx]
        configs.append(cfg)
    return configs


def screen(
    run_fn: Callable[[dict[str, Any]], dict[str, float]],
    factors: list[Factor],
    *,
    method: str = "morris",
    n_trajectories: int = 100,
    seed: int = 42,
) -> dict[str, NDArray[np.floating[Any]]]:
    """Screen factors for influence on observables via SALib.

    Args:
        run_fn: Callable that takes a config dict and returns a dict of
            observable name → scalar score.
        factors: List of continuous factors to screen.
        method: Screening method ("morris" or "sobol").
        n_trajectories: Number of Morris trajectories or Sobol samples.
        seed: Random seed.

    Returns:
        Dictionary mapping observable names to arrays of factor importance
        (mu_star for Morris, S1 for Sobol), one value per factor.

    Raises:
        NotImplementedError: If method is not "morris".
        ValueError: If no continuous factors are provided.
    """
    from SALib.analyze import morris as morris_analyze  # type: ignore[import-untyped]
    from SALib.sample import morris as morris_sample  # type: ignore[import-untyped]

    if method != "morris":
        msg = f"Screening method {method!r} not yet implemented"
        raise NotImplementedError(msg)

    continuous = [f for f in factors if f.factor_type == FactorType.CONTINUOUS]
    if not continuous:
        msg = "Screening requires at least one continuous factor"
        raise ValueError(msg)

    problem: dict[str, Any] = {
        "num_vars": len(continuous),
        "names": [f.name for f in continuous],
        "bounds": [list(f.bounds) for f in continuous if f.bounds is not None],
    }
    param_values = morris_sample.sample(problem, n_trajectories, seed=seed)

    # Evaluate model at each sample point
    results_by_obs: dict[str, list[float]] = {}
    for row in param_values:
        cfg = dict(zip(problem["names"], row, strict=True))
        scores = run_fn(cfg)
        for obs_name, val in scores.items():
            results_by_obs.setdefault(obs_name, []).append(val)

    importance: dict[str, NDArray[np.floating[Any]]] = {}
    for obs_name, vals in results_by_obs.items():
        si = morris_analyze.analyze(
            problem,
            param_values,
            np.array(vals),
            seed=seed,
        )
        importance[obs_name] = np.asarray(si["mu_star"], dtype=np.float64)

    return importance


def reduce_factors(
    factors: list[Factor],
    importance: dict[str, NDArray[np.floating[Any]]],
    *,
    threshold: float = 0.1,
) -> list[Factor]:
    """Keep only factors whose max importance exceeds threshold.

    Args:
        factors: Original factor list.
        importance: Output of ``screen()``.
        threshold: Minimum importance to retain a factor.

    Returns:
        Reduced list of influential factors.
    """
    continuous = [f for f in factors if f.factor_type == FactorType.CONTINUOUS]
    non_continuous = [f for f in factors if f.factor_type != FactorType.CONTINUOUS]

    max_importance = np.zeros(len(continuous))
    for arr in importance.values():
        max_importance = np.maximum(max_importance, arr)

    kept = [
        f for f, imp in zip(continuous, max_importance, strict=True) if imp >= threshold
    ]
    return non_continuous + kept
