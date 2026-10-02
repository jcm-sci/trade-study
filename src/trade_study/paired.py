"""Paired comparisons between designs under common random numbers (#133).

When a simulator derives each dataset from the replicate index alone, every
design sees the same datasets, so differences between designs are best
estimated replicate by replicate. These helpers match rows of a
``ResultsTable`` by ``metadata["rep"]`` (as written by ``run_grid`` with
``n_reps > 1``) and summarize the per-replicate differences.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .protocols import ResultsTable

Design = Mapping[str, Any] | int
METHODS = ("bootstrap", "t")


@dataclass(frozen=True)
class PairedDifference:
    """Mean per-replicate difference ``a - b`` with a confidence interval.

    Attributes:
        design_a: Config of the first design.
        design_b: Config of the second design.
        observable: Observable compared.
        mean: Mean of the per-replicate differences.
        lower: Lower confidence bound.
        upper: Upper confidence bound.
        n_pairs: Replicates with a finite value for both designs.
        confidence: Confidence level of the interval.
        method: ``"bootstrap"`` (percentile) or ``"t"`` (paired t).
    """

    design_a: dict[str, Any]
    design_b: dict[str, Any]
    observable: str
    mean: float
    lower: float
    upper: float
    n_pairs: int
    confidence: float
    method: str


def paired_difference(  # ruff: ignore[too-many-arguments]
    results: ResultsTable,
    design_a: Design,
    design_b: Design,
    observable: str,
    *,
    method: str = "bootstrap",
    confidence: float = 0.95,
    n_boot: int = 2000,
    seed: int = 0,
) -> PairedDifference:
    """Estimate the mean difference ``a - b`` over shared replicates.

    Args:
        results: Per-replicate results; every row needs ``metadata["rep"]``.
        design_a: First design, as a config subset that identifies one design
            point or as its ``metadata["design_point"]`` index.
        design_b: Second design, specified the same way.
        observable: Name of the observable to compare.
        method: ``"bootstrap"`` for a percentile bootstrap over replicates,
            or ``"t"`` for a paired t interval (requires scipy).
        confidence: Two-sided confidence level.
        n_boot: Bootstrap resamples.
        seed: Bootstrap seed.

    Returns:
        The paired difference and its interval. Replicates where either
        design's value is non-finite are dropped and not counted in
        ``n_pairs``.

    Raises:
        ValueError: If the designs' replicate sets differ, a design repeats a
            replicate, fewer than two finite pairs remain, or ``method`` is
            unknown.
    """
    if method not in METHODS:
        msg = f"method must be one of {METHODS}, got {method!r}"
        raise ValueError(msg)
    rows_a, config_a = _select(results, design_a)
    rows_b, config_b = _select(results, design_b)
    diff = _paired_differences(
        _column(results, observable),
        _by_rep(results, rows_a, config_a),
        _by_rep(results, rows_b, config_b),
    )
    lower, upper = (
        _bootstrap_interval(diff, confidence, n_boot, seed)
        if method == "bootstrap"
        else _t_interval(diff, confidence)
    )
    return PairedDifference(
        design_a=config_a,
        design_b=config_b,
        observable=observable,
        mean=float(diff.mean()),
        lower=lower,
        upper=upper,
        n_pairs=len(diff),
        confidence=confidence,
        method=method,
    )


def paired_rank(  # ruff: ignore[too-many-arguments]
    results: ResultsTable,
    observable: str,
    reference: Design,
    *,
    maximize: bool = False,
    adjust: str | None = "bonferroni",
    method: str = "bootstrap",
    confidence: float = 0.95,
    n_boot: int = 2000,
    seed: int = 0,
) -> list[PairedDifference]:
    """Compare every design with a reference design, best first.

    Each entry is the paired difference ``design - reference``. With
    ``adjust="bonferroni"`` each interval uses confidence
    ``1 - (1 - confidence) / m`` for ``m`` comparisons, so all intervals hold
    jointly at ``confidence``; with ``adjust=None`` they hold one at a time.

    Args:
        results: Per-replicate results; every row needs ``metadata["rep"]``.
        observable: Name of the observable to compare.
        reference: Reference design (config subset or design-point index).
        maximize: Whether larger values are better (sets the ordering).
        adjust: ``"bonferroni"`` or ``None``.
        method: Interval method, as in :func:`paired_difference`.
        confidence: Joint (adjusted) or per-comparison confidence level.
        n_boot: Bootstrap resamples.
        seed: Bootstrap seed.

    Returns:
        One paired difference per non-reference design, ordered from the
        most improved on the reference to the least.

    Raises:
        ValueError: If ``adjust`` is unknown.
    """
    if adjust not in {"bonferroni", None}:
        msg = f"adjust must be 'bonferroni' or None, got {adjust!r}"
        raise ValueError(msg)
    _rows, reference_config = _select(results, reference)
    others = [
        config
        for config in _design_configs(results)
        if _key(config) != _key(reference_config)
    ]
    level = confidence
    if adjust == "bonferroni" and others:
        level = 1 - (1 - confidence) / len(others)
    ranked = [
        paired_difference(
            results,
            config,
            reference_config,
            observable,
            method=method,
            confidence=level,
            n_boot=n_boot,
            seed=seed,
        )
        for config in others
    ]
    return sorted(ranked, key=lambda d: -d.mean if maximize else d.mean)


def _column(results: ResultsTable, observable: str) -> NDArray[np.floating[Any]]:
    if observable not in results.observable_names:
        msg = f"unknown observable {observable!r}"
        raise KeyError(msg)
    column: NDArray[np.floating[Any]] = results.scores[
        :, results.observable_names.index(observable)
    ]
    return column


def _key(config: Mapping[str, Any]) -> str:
    return json.dumps(dict(config), sort_keys=True, default=repr)


def _design_configs(results: ResultsTable) -> list[dict[str, Any]]:
    """Return each distinct design's config, in first-seen order.

    Returns:
        Distinct configs.
    """
    seen: dict[str, dict[str, Any]] = {}
    for config in results.configs:
        seen.setdefault(_key(config), dict(config))
    return list(seen.values())


def _select(results: ResultsTable, design: Design) -> tuple[list[int], dict[str, Any]]:
    """Return the rows of one design and its full config.

    Returns:
        Row indices and the design's config.

    Raises:
        ValueError: If the design matches no rows or more than one design.
    """
    if isinstance(design, int):
        rows = [
            i
            for i, meta in enumerate(results.metadata)
            if meta.get("design_point") == design
        ]
    else:
        rows = [
            i
            for i, config in enumerate(results.configs)
            if all(k in config and config[k] == v for k, v in design.items())
        ]
    if not rows:
        msg = f"no rows match design {design!r}"
        raise ValueError(msg)
    configs = {_key(results.configs[i]) for i in rows}
    if len(configs) > 1:
        msg = f"design {design!r} matches {len(configs)} distinct configs"
        raise ValueError(msg)
    return rows, dict(results.configs[rows[0]])


def _by_rep(
    results: ResultsTable, rows: list[int], config: dict[str, Any]
) -> dict[Any, int]:
    """Map each replicate index to its row.

    Returns:
        Replicate index to row index.

    Raises:
        ValueError: If a row lacks ``rep`` metadata or a replicate repeats.
    """
    by_rep: dict[Any, int] = {}
    for row in rows:
        meta = results.metadata[row] if results.metadata else {}
        if "rep" not in meta:
            msg = "every row needs metadata['rep'] to pair replicates"
            raise ValueError(msg)
        if meta["rep"] in by_rep:
            msg = f"design {config!r} repeats replicate {meta['rep']!r}"
            raise ValueError(msg)
        by_rep[meta["rep"]] = row
    return by_rep


def _paired_differences(
    column: NDArray[np.floating[Any]],
    by_rep_a: dict[Any, int],
    by_rep_b: dict[Any, int],
) -> NDArray[np.floating[Any]]:
    """Return the finite per-replicate differences ``a - b``.

    Returns:
        Differences ordered by replicate.

    Raises:
        ValueError: If the replicate sets differ or fewer than two finite
            pairs remain.
    """
    if by_rep_a.keys() != by_rep_b.keys():
        only_a = sorted(by_rep_a.keys() - by_rep_b.keys())
        only_b = sorted(by_rep_b.keys() - by_rep_a.keys())
        msg = (
            f"designs have different replicates: only in a {only_a}, only in b {only_b}"
        )
        raise ValueError(msg)
    reps = sorted(by_rep_a)
    a = np.array([column[by_rep_a[r]] for r in reps], dtype=float)
    b = np.array([column[by_rep_b[r]] for r in reps], dtype=float)
    diff: NDArray[np.floating[Any]] = (a - b)[np.isfinite(a) & np.isfinite(b)]
    if len(diff) < 2:
        msg = f"need at least two finite replicate pairs, got {len(diff)}"
        raise ValueError(msg)
    return diff


def _bootstrap_interval(
    diff: NDArray[np.floating[Any]], confidence: float, n_boot: int, seed: int
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    means = diff[draws].mean(axis=1)
    tail = (1 - confidence) / 2
    lower, upper = np.quantile(means, [tail, 1 - tail])
    return float(lower), float(upper)


def _t_interval(
    diff: NDArray[np.floating[Any]], confidence: float
) -> tuple[float, float]:
    try:
        from scipy import stats  # type: ignore[import-untyped]
    except ImportError as exc:
        msg = "method='t' requires scipy; install trade-study[design]"
        raise ImportError(msg) from exc
    n = len(diff)
    half = float(stats.t.ppf(0.5 + confidence / 2, n - 1)) * diff.std(ddof=1)
    half /= np.sqrt(n)
    mean = float(diff.mean())
    return mean - half, mean + half
