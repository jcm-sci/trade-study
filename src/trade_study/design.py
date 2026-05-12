"""Experimental design and factor screening.

Wraps pyDOE3 for grid construction and SALib for sensitivity screening.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from itertools import product
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from numpy.typing import NDArray

# Maximum oversample factor when rejection-sampling a constrained QMC/LHS
# design before giving up.
_MAX_OVERSAMPLE: int = 64
# Warn when the realized feasibility ratio falls below this fraction; below
# this, QMC space-filling guarantees degrade noticeably.
_LOW_FEASIBILITY_WARN: float = 0.1


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
            ValueError: If name is empty, continuous factor has missing or
                invalid bounds, or discrete/categorical factor has empty
                levels.
        """
        if not self.name:
            msg = "Factor name must be a non-empty string"
            raise ValueError(msg)
        if self.factor_type == FactorType.CONTINUOUS:
            if self.bounds is None:
                msg = f"Continuous factor '{self.name}' requires bounds"
                raise ValueError(msg)
            lo, hi = self.bounds
            if not (np.isfinite(lo) and np.isfinite(hi)):
                msg = f"Continuous factor '{self.name}' bounds must be finite"
                raise ValueError(msg)
            if lo >= hi:
                msg = f"Continuous factor '{self.name}' requires lo < hi"
                raise ValueError(msg)
        else:
            if self.levels is None:
                msg = f"Factor '{self.name}' of type {self.factor_type} requires levels"
                raise ValueError(msg)
            if len(self.levels) == 0:
                msg = f"Factor '{self.name}' levels must be non-empty"
                raise ValueError(msg)


@dataclass(frozen=True)
class FactorConstraint:
    """Coupled-factor constraint applied at design-generation time.

    Unlike :class:`trade_study.protocols.Constraint`, which filters trials
    after simulation based on observables, a ``FactorConstraint`` filters
    candidate configurations *before* they are evaluated. It is the way
    to express coupled constraints such as "if ``method == 'a'`` then
    ``patience <= 2``" without distorting the design via post-hoc
    filtering of returned results.

    Attributes:
        predicate: Callable taking a candidate config dict and returning
            ``True`` when the config is feasible.
        name: Human-readable label used in error messages and warnings.
    """

    predicate: Callable[[dict[str, Any]], bool]
    name: str = field(default="factor_constraint")

    def is_feasible(self, config: dict[str, Any]) -> bool:
        """Return whether ``config`` satisfies this constraint.

        Args:
            config: Candidate design point.

        Returns:
            ``True`` if the predicate accepts the config.
        """
        return bool(self.predicate(config))


def _all_feasible(
    config: dict[str, Any],
    constraints: Sequence[FactorConstraint] | None,
) -> bool:
    """Return whether ``config`` satisfies every constraint.

    Args:
        config: Candidate design point.
        constraints: Constraints to check; ``None`` is treated as no
            constraints.

    Returns:
        ``True`` if all constraints accept ``config`` (or no constraints
        were supplied).
    """
    if not constraints:
        return True
    return all(c.is_feasible(config) for c in constraints)


def build_grid(
    factors: list[Factor],
    *,
    method: str = "full",
    n_samples: int = 100,
    seed: int = 42,
    scramble: bool = True,
    constraints: Sequence[FactorConstraint] | None = None,
) -> list[dict[str, Any]]:
    """Build an experimental design grid.

    Args:
        factors: List of design factors.
        method: Design method. One of:
            - "full": Full factorial (categorical/discrete only).
            - "lhs": Latin hypercube sampling (continuous factors, maps
              categorical factors to uniform random selection).
            - "sobol": Scrambled Sobol' sequence via ``scipy.stats.qmc``.
            - "halton": Scrambled Halton sequence via ``scipy.stats.qmc``.
        n_samples: Number of samples for LHS / QMC methods.
        seed: Random seed.
        scramble: Whether to apply scrambling to QMC sequences (Sobol /
            Halton). Ignored for other methods.
        constraints: Optional sequence of :class:`FactorConstraint`
            applied at design-generation time. For ``"full"``, the
            Cartesian product is filtered. For ``"lhs"``/``"sobol"``/
            ``"halton"``, rejection sampling with oversampling is used
            to return ``n_samples`` feasible configs (rejection breaks
            the QMC space-filling guarantee; expect higher discrepancy
            when the feasibility ratio is low).

    Returns:
        List of config dictionaries, one per design point.

    Raises:
        ValueError: If an unknown design method is specified, or if
            constraints reject more than the maximum oversample budget
            allows.
    """
    if method == "full":
        return _full_factorial(factors, constraints=constraints)
    if method == "lhs":
        return _latin_hypercube(
            factors,
            n_samples=n_samples,
            seed=seed,
            constraints=constraints,
        )
    if method in {"sobol", "halton"}:
        return _qmc_sample(
            factors,
            n_samples=n_samples,
            seed=seed,
            qmc_method=method,
            scramble=scramble,
            constraints=constraints,
        )
    msg = f"Unknown design method: {method!r}"
    raise ValueError(msg)


def _full_factorial(
    factors: list[Factor],
    *,
    constraints: Sequence[FactorConstraint] | None = None,
) -> list[dict[str, Any]]:
    """Full factorial over all factor levels, optionally constrained.

    Args:
        factors: Categorical or discrete factors.
        constraints: Optional design-time constraints to filter the
            Cartesian product.

    Returns:
        List of config dictionaries, one per design point. Empty if
        every combination is rejected by the constraints.

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
    configs = (dict(zip(names, combo, strict=True)) for combo in product(*level_lists))
    return [c for c in configs if _all_feasible(c, constraints)]


def _row_to_config(
    row: NDArray[np.floating[Any]], factors: list[Factor]
) -> dict[str, Any]:
    """Map a unit-cube QMC/LHS row into a factor-keyed config dict.

    Args:
        row: Array in [0, 1) per dimension, one entry per factor.
        factors: Factor list defining each dimension.

    Returns:
        Config dictionary keyed by factor name.
    """
    cfg: dict[str, Any] = {}
    for j, f in enumerate(factors):
        if f.factor_type == FactorType.CONTINUOUS and f.bounds is not None:
            lo, hi = f.bounds
            cfg[f.name] = lo + float(row[j]) * (hi - lo)
        elif f.levels is not None:
            idx = int(row[j] * len(f.levels))
            idx = min(idx, len(f.levels) - 1)
            cfg[f.name] = f.levels[idx]
    return cfg


def _latin_hypercube(
    factors: list[Factor],
    *,
    n_samples: int,
    seed: int,
    constraints: Sequence[FactorConstraint] | None = None,
) -> list[dict[str, Any]]:
    """Latin hypercube design via pyDOE3, with optional rejection.

    Args:
        factors: Design factors.
        n_samples: Number of feasible samples to return.
        seed: Random seed.
        constraints: Optional design-time constraints; rejected samples
            are discarded and additional LHS batches are drawn until
            ``n_samples`` feasible configs are collected (or the
            oversample budget is exhausted).

    Returns:
        List of feasible config dictionaries with length ``n_samples``.
        Propagates :class:`ValueError` from the rejection sampler when
        the feasibility ratio is too low.
    """
    from pyDOE3 import lhs  # type: ignore[import-untyped]

    n_factors = len(factors)
    if not constraints:
        raw = lhs(n_factors, samples=n_samples, criterion="maximin", seed=seed)
        return [_row_to_config(row, factors) for row in raw]

    return _rejection_sample(
        n_samples=n_samples,
        constraints=constraints,
        method_name="lhs",
        draw=lambda batch_size, batch_seed: lhs(
            n_factors,
            samples=batch_size,
            criterion="maximin",
            seed=batch_seed,
        ),
        factors=factors,
        seed=seed,
    )


def _qmc_sample(
    factors: list[Factor],
    *,
    n_samples: int,
    seed: int,
    qmc_method: str,
    scramble: bool,
    constraints: Sequence[FactorConstraint] | None = None,
) -> list[dict[str, Any]]:
    """Quasi-Monte Carlo design via ``scipy.stats.qmc``.

    Args:
        factors: List of design factors.
        n_samples: Number of sample points.
        seed: Random seed for scrambling.
        qmc_method: ``"sobol"`` or ``"halton"``.
        scramble: Whether to apply scrambling.
        constraints: Optional design-time constraints. When supplied,
            the QMC sampler is advanced as a single long sequence and
            infeasible points are skipped, preserving the ordering of
            the underlying low-discrepancy sequence among accepted
            points (rejection still degrades the realized discrepancy).

    Returns:
        List of config dictionaries with length ``n_samples``.
        Propagates :class:`ValueError` from the rejection sampler when
        the feasibility ratio is too low.
    """
    from scipy.stats import qmc  # type: ignore[import-untyped]

    n_factors = len(factors)
    sampler: qmc.QMCEngine
    if qmc_method == "sobol":
        sampler = qmc.Sobol(d=n_factors, scramble=scramble, seed=seed)
    else:
        sampler = qmc.Halton(d=n_factors, scramble=scramble, seed=seed)

    if not constraints:
        raw = sampler.random(n_samples)
        return [_row_to_config(row, factors) for row in raw]

    return _rejection_sample(
        n_samples=n_samples,
        constraints=constraints,
        method_name=qmc_method,
        draw=lambda batch_size, _seed: sampler.random(batch_size),
        factors=factors,
        seed=seed,
    )


def _rejection_sample(
    *,
    n_samples: int,
    constraints: Sequence[FactorConstraint],
    method_name: str,
    draw: Callable[[int, int], NDArray[np.floating[Any]]],
    factors: list[Factor],
    seed: int,
) -> list[dict[str, Any]]:
    """Collect ``n_samples`` feasible configs via rejection sampling.

    Doubles the oversample factor on each retry until the budget is
    exhausted. Emits a :class:`UserWarning` when the realized
    feasibility ratio falls below :data:`_LOW_FEASIBILITY_WARN`.

    Args:
        n_samples: Number of feasible configs to return.
        constraints: Constraints applied to each candidate.
        method_name: Name of the sampler (for error messages).
        draw: Callable taking ``(batch_size, seed)`` and returning a
            ``(batch_size, n_factors)`` unit-cube array.
        factors: Factor list used to map rows to configs.
        seed: Seed forwarded to ``draw`` (sequence-based samplers may
            ignore it after the first call).

    Returns:
        Exactly ``n_samples`` feasible configs.

    Raises:
        ValueError: If the constraints reject too many candidates to
            reach ``n_samples`` within :data:`_MAX_OVERSAMPLE` x the
            requested size.
    """
    import warnings

    accepted: list[dict[str, Any]] = []
    drawn = 0
    oversample = 2
    while len(accepted) < n_samples and oversample <= _MAX_OVERSAMPLE:
        deficit = n_samples - len(accepted)
        batch_size = max(deficit * oversample, 1)
        rows = draw(batch_size, seed + drawn)
        drawn += batch_size
        for row in rows:
            cfg = _row_to_config(row, factors)
            if _all_feasible(cfg, constraints):
                accepted.append(cfg)
                if len(accepted) == n_samples:
                    break
        oversample *= 2

    if len(accepted) < n_samples:
        ratio = len(accepted) / max(drawn, 1)
        msg = (
            f"{method_name}: constraints rejected too many candidates "
            f"({len(accepted)}/{n_samples} feasible after drawing {drawn}; "
            f"feasibility ratio {ratio:.3g}). Loosen constraints or "
            f"reduce n_samples."
        )
        raise ValueError(msg)

    ratio = len(accepted) / drawn
    if ratio < _LOW_FEASIBILITY_WARN:
        warnings.warn(
            f"{method_name}: low feasibility ratio {ratio:.3g} "
            f"({len(accepted)}/{drawn}); QMC space-filling properties "
            f"are degraded by rejection.",
            UserWarning,
            stacklevel=3,
        )
    return accepted


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
        method: Screening method (``"morris"`` or ``"sobol"``).
        n_trajectories: Number of Morris trajectories.  For Sobol, this
            controls the base sample size *N*; the total number of model
            evaluations is *N* x (num_vars + 2).
        seed: Random seed.

    Returns:
        Dictionary mapping observable names to arrays of factor importance
        (mu_star for Morris, S1 for Sobol), one value per factor.

    Raises:
        ValueError: If *method* is unknown or no continuous factors are
            provided.
    """
    continuous = [f for f in factors if f.factor_type == FactorType.CONTINUOUS]
    if not continuous:
        msg = "Screening requires at least one continuous factor"
        raise ValueError(msg)

    problem: dict[str, Any] = {
        "num_vars": len(continuous),
        "names": [f.name for f in continuous],
        "bounds": [list(f.bounds) for f in continuous if f.bounds is not None],
    }

    if method == "morris":
        return _screen_morris(run_fn, problem, n_trajectories, seed)
    if method == "sobol":
        return _screen_sobol(run_fn, problem, n_trajectories, seed)

    msg = f"Unknown screening method: {method!r}"
    raise ValueError(msg)


def _screen_morris(
    run_fn: Callable[[dict[str, Any]], dict[str, float]],
    problem: dict[str, Any],
    n_trajectories: int,
    seed: int,
) -> dict[str, NDArray[np.floating[Any]]]:
    """Morris elementary-effects screening.

    Returns:
        Mapping from observable name to mu_star array.
    """
    from SALib.analyze import morris as morris_analyze  # type: ignore[import-untyped]
    from SALib.sample import morris as morris_sample  # type: ignore[import-untyped]

    param_values = morris_sample.sample(problem, n_trajectories, seed=seed)

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


def _screen_sobol(
    run_fn: Callable[[dict[str, Any]], dict[str, float]],
    problem: dict[str, Any],
    n_samples: int,
    seed: int,
) -> dict[str, NDArray[np.floating[Any]]]:
    """Sobol variance-based sensitivity analysis.

    Returns:
        Mapping from observable name to S1 (first-order) index array.
    """
    from SALib.analyze import sobol as sobol_analyze
    from SALib.sample import sobol as sobol_sample

    param_values = sobol_sample.sample(problem, n_samples, seed=seed)

    results_by_obs: dict[str, list[float]] = {}
    for row in param_values:
        cfg = dict(zip(problem["names"], row, strict=True))
        scores = run_fn(cfg)
        for obs_name, val in scores.items():
            results_by_obs.setdefault(obs_name, []).append(val)

    importance: dict[str, NDArray[np.floating[Any]]] = {}
    for obs_name, vals in results_by_obs.items():
        si = sobol_analyze.analyze(
            problem,
            np.array(vals),
            seed=seed,
        )
        importance[obs_name] = np.asarray(si["S1"], dtype=np.float64)

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
