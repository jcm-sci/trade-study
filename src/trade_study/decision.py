"""Preference sensitivity and practical equivalence over collected results."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from .paired import paired_rank
from .protocols import Direction, ResultsTable

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .paired import PairedDifference
    from .protocols import Constraint, Observable


@dataclass(frozen=True)
class PreferencePolicy:
    """Explicit normalization, preferences, equivalence, and paired assumptions.

    Attributes:
        weights: Preference vectors by observable name; nonnegative weights
            are normalized to sum to one. Omitted objectives have zero weight.
        normalization: ``minmax`` over finite feasible design means,
            ``reference`` using reference_bounds, or ``none`` in raw units.
        reference_bounds: Raw-unit lower/upper anchors for reference scaling.
        equivalence: Pairwise practical-equivalence tolerances in raw units;
            omitted observables require exact equality.
        paired_confidence: Nominal joint confidence across requested comparisons
            and observables. Bootstrap intervals are approximate.
        paired_method: Existing paired-comparison method, ``bootstrap`` or ``t``.
        n_boot: Bootstrap resample count.
        seed: Bootstrap seed; preference vectors themselves are explicit.
    """

    weights: list[dict[str, float]]
    normalization: str
    reference_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    equivalence: dict[str, float] = field(default_factory=dict)
    paired_confidence: float = 0.95
    paired_method: str = "bootstrap"
    n_boot: int = 2000
    seed: int = 0


@dataclass(frozen=True)
class PreferenceSweep:
    """Exportable decision summaries and rankings under explicit preferences.

    Attributes:
        summary: One raw-mean row per design, with decision metadata;
            export using ResultsTable.to_dataframe(include_metadata=True).
        weights: Effective normalized weights, preference by observable.
        utilities: Direction-aware weighted losses, preference by design.
        ranks: Competition ranks (ties share a rank); excluded designs are NaN.
        feasible: Finite designs meeting the supplied constraints.
        pareto: Feasible nondominated designs, independently of preferences.
        equivalent: Pairwise raw-unit practical-equivalence matrix.
        normalization_bounds: Actual anchors; empty for raw-unit normalization.
        paired: Optional existing paired comparisons against a supplied reference.
        policy: Copy of all requested preference and inference assumptions.
    """

    summary: ResultsTable
    weights: NDArray[np.float64]
    utilities: NDArray[np.float64]
    ranks: NDArray[np.float64]
    feasible: NDArray[np.bool_]
    pareto: NDArray[np.bool_]
    equivalent: NDArray[np.bool_]
    normalization_bounds: dict[str, tuple[float, float]]
    paired: dict[str, list[PairedDifference]]
    policy: PreferencePolicy


def _design_table(results: ResultsTable) -> ResultsTable:
    if results.metadata and len(results.metadata) != len(results.configs):
        msg = "Decision results require row-aligned metadata"
        raise ValueError(msg)
    if not any("rep" in meta for meta in results.metadata):
        return deepcopy(results)
    if not all("rep" in meta and "design_point" in meta for meta in results.metadata):
        msg = "Replicated decisions require complete design_point/rep identities"
        raise ValueError(msg)
    seen: set[tuple[int, int]] = set()
    configs: dict[int, dict[str, Any]] = {}
    for config, meta in zip(results.configs, results.metadata, strict=True):
        identity = meta["design_point"], meta["rep"]
        if identity in seen or (
            identity[0] in configs and config != configs[identity[0]]
        ):
            msg = "Duplicate replicate or conflicting design-point configuration"
            raise ValueError(msg)
        seen.add(identity)
        configs[identity[0]] = config
    return results.aggregate_replicates()


def _weights(policy: PreferencePolicy, names: list[str]) -> NDArray[np.float64]:
    if not policy.weights or any(set(w) - set(names) for w in policy.weights):
        msg = "Preference weights require nonempty vectors of known observables"
        raise ValueError(msg)
    weights = np.array(
        [[w.get(name, 0) for name in names] for w in policy.weights], dtype=float
    )
    if (
        not np.all(np.isfinite(weights))
        or np.any(weights < 0)
        or np.any(weights.max(axis=1) <= 0)
    ):
        msg = "Preference weights must be finite, nonnegative and have positive totals"
        raise ValueError(msg)
    weights /= weights.max(axis=1, keepdims=True)
    weights /= weights.sum(axis=1, keepdims=True)
    if set(policy.equivalence) - set(names) or any(
        not np.isfinite(v) or v < 0 for v in policy.equivalence.values()
    ):
        msg = "Equivalence requires nonnegative finite tolerances for known observables"
        raise ValueError(msg)
    return weights


def _normalize(
    raw: NDArray[np.float64],
    eligible: NDArray[np.bool_],
    names: list[str],
    policy: PreferencePolicy,
) -> tuple[NDArray[np.float64], dict[str, tuple[float, float]]]:
    if policy.normalization == "none":
        return raw.copy(), {}
    if policy.normalization == "minmax":
        bounds = (
            {
                name: (float(raw[eligible, j].min()), float(raw[eligible, j].max()))
                for j, name in enumerate(names)
            }
            if np.any(eligible)
            else dict.fromkeys(names, (np.nan, np.nan))
        )
    elif policy.normalization == "reference":
        if set(policy.reference_bounds) != set(names):
            msg = "reference_bounds must specify every decision observable exactly"
            raise ValueError(msg)
        bounds = dict(policy.reference_bounds)
        for name, anchors in bounds.items():
            if (
                len(anchors) != 2
                or not np.all(np.isfinite(anchors))
                or anchors[0] > anchors[1]
            ):
                msg = f"Invalid normalization bounds for {name!r}"
                raise ValueError(msg)
    else:
        msg = "normalization must be 'minmax', 'reference', or 'none'"
        raise ValueError(msg)
    normalized = np.full(raw.shape, np.nan)
    for j, name in enumerate(names):
        lower, upper = bounds[name]
        if upper == lower:
            if np.any(raw[eligible, j] != lower):
                msg = f"Zero-range reference bounds do not describe {name!r}"
                raise ValueError(msg)
            normalized[eligible, j] = 0
        else:
            normalized[eligible, j] = _scaled_values(raw[eligible, j], lower, upper)
    return normalized, bounds


def _scaled_values(
    values: NDArray[np.float64], lower: float, upper: float
) -> NDArray[np.float64]:
    if values.size and not np.isfinite(upper - lower):
        msg = "Normalization exceeds finite floating range; rescale observable units"
        raise ValueError(msg)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            return (values - lower) / (upper - lower)
    except FloatingPointError as error:
        msg = "Normalization exceeds finite floating range; rescale observable units"
        raise ValueError(msg) from error


def _ranks(
    utilities: NDArray[np.float64],
    eligible: NDArray[np.bool_],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    ranks = np.full(utilities.shape, np.nan)
    selection = np.zeros(utilities.shape[1])
    if np.any(eligible):
        for i, losses in enumerate(utilities[:, eligible]):
            ranks[i, eligible] = (
                np.searchsorted(np.sort(losses), losses, side="left") + 1
            )
            winners = ranks[i] == 1
            selection[winners] += 1 / winners.sum()
        selection /= len(utilities)
    return ranks, selection


def _alternatives(
    raw: NDArray[np.float64],
    eligible: NDArray[np.bool_],
    observables: list[Observable],
    policy: PreferencePolicy,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    oriented = raw * np.array([
        1 if o.direction == Direction.MINIMIZE else -1 for o in observables
    ])
    pareto = np.zeros(len(raw), dtype=bool)
    for index in np.flatnonzero(eligible):
        pareto[index] = not np.any(
            np.all(oriented[eligible] <= oriented[index], axis=1)
            & np.any(oriented[eligible] < oriented[index], axis=1)
        )
    equivalent = eligible[:, None] & eligible[None, :]
    safe = np.where(np.isfinite(raw), raw, 0)
    for j, observable in enumerate(observables):
        equivalent &= np.abs(
            safe[:, j, None] - safe[None, :, j]
        ) <= policy.equivalence.get(observable.name, 0)
    return pareto, equivalent


def _paired(
    results: ResultsTable,
    designs: ResultsTable,
    eligible: NDArray[np.bool_],
    observables: list[Observable],
    reference: int | dict[str, Any],
    policy: PreferencePolicy,
) -> dict[str, list[PairedDifference]]:
    if not results.metadata or not all(
        "rep" in m and "design_point" in m for m in results.metadata
    ):
        msg = "Paired decision summaries require raw design_point/rep observations"
        raise ValueError(msg)
    ids = {
        meta["design_point"]
        for meta, keep in zip(designs.metadata, eligible, strict=True)
        if keep
    }
    rows = [i for i, meta in enumerate(results.metadata) if meta["design_point"] in ids]
    filtered = ResultsTable(
        [results.configs[i] for i in rows],
        results.scores[rows],
        results.observable_names,
        metadata=[results.metadata[i] for i in rows],
    )
    level = 1 - (1 - policy.paired_confidence) / len(observables)
    return {
        o.name: paired_rank(
            filtered,
            o.name,
            reference,
            maximize=o.direction == Direction.MAXIMIZE,
            confidence=level,
            method=policy.paired_method,
            n_boot=policy.n_boot,
            seed=policy.seed,
        )
        for o in observables
    }


def _decorate_summary(report: PreferenceSweep, selection: NDArray[np.float64]) -> None:
    for i, meta in enumerate(report.summary.metadata):
        valid = bool(report.feasible[i])
        regrets = (
            report.utilities[:, i] - np.nanmin(report.utilities, axis=1)
            if valid
            else np.array([np.nan])
        )
        meta["decision"] = {
            "feasible": valid,
            "pareto": bool(report.pareto[i]),
            "selection_fraction": float(selection[i]),
            "best_rank": float(report.ranks[:, i].min()) if valid else np.nan,
            "worst_rank": float(report.ranks[:, i].max()) if valid else np.nan,
            "mean_rank": float(report.ranks[:, i].mean()) if valid else np.nan,
            "max_regret": float(regrets.max()),
            "equivalent_to": np.flatnonzero(report.equivalent[i]).tolist(),
            "normalization": report.policy.normalization,
            "normalization_bounds": report.normalization_bounds,
            "preferences_evaluated": len(report.weights),
        }


def _decision_table(
    results: ResultsTable, names: list[str], constraints: list[Constraint]
) -> tuple[ResultsTable, NDArray[np.bool_]]:
    designs = _design_table(results)
    raw = np.asarray(
        designs.scores[:, [designs.observable_names.index(n) for n in names]],
        dtype=float,
    ).copy()
    for i, meta in enumerate(designs.metadata):
        for j, name in enumerate(names):
            raw[i, j] = meta.get("scores", {}).get(name, raw[i, j])
    eligible = designs.feasible(constraints) & np.all(np.isfinite(raw), axis=1)
    table = ResultsTable(
        deepcopy(designs.configs),
        raw,
        names,
        annotations=designs.annotations,
        annotation_names=designs.annotation_names,
        metadata=deepcopy(designs.metadata)
        if designs.metadata
        else [{} for _ in designs.configs],
    )
    return table, eligible


def _utilities(
    normalized: NDArray[np.float64],
    eligible: NDArray[np.bool_],
    weights: NDArray[np.float64],
    observables: list[Observable],
) -> NDArray[np.float64]:
    signs = np.array([
        1 if o.direction == Direction.MINIMIZE else -1 for o in observables
    ])
    utilities = np.full((len(weights), len(normalized)), np.nan)
    try:
        with np.errstate(over="raise", invalid="raise"):
            utilities[:, eligible] = weights @ (normalized[eligible] * signs).T
    except FloatingPointError as error:
        msg = "Weighted loss exceeds finite floating range; rescale observable units"
        raise ValueError(msg) from error
    return utilities


def preference_sweep(
    results: ResultsTable,
    observables: list[Observable],
    *,
    policy: PreferencePolicy,
    constraints: list[Constraint] | None = None,
    paired_reference: int | dict[str, Any] | None = None,
) -> PreferenceSweep:
    """Assess choices across explicit preferences without evaluating a simulator.

    Args:
        results: Existing raw-replicate or per-design results. Session raw means
            in metadata override weighted score columns.
        observables: Decision objectives with directions. Observable.weight is
            replaced by the explicit preference vectors, avoiding double weights.
        policy: Explicit normalization, preferences and equivalence assumptions.
        constraints: Feasibility restrictions, using existing ResultsTable rules.
        paired_reference: Optional raw design-point id or config selector for
            existing paired comparisons. Requires matching replicate sets.

    Returns:
        Per-design exportable summaries, ranks and regret across preferences,
        feasible Pareto alternatives, pairwise practical equivalence and optional
        paired uncertainty. Tied winners share selection credit. With no finite
        feasible alternatives, selection fractions are zero and ranks are NaN.

    Raises:
        ValueError: If objective names, table shape, preferences, normalization,
            or paired inference assumptions are invalid.

    Notes:
        Minmax anchors depend on this feasible set. Reference anchors do not clip
        values outside their range. Practical equivalence compares raw means,
        not statistical evidence. Paired bootstrap/t assumptions and approximate
        coverage remain those of existing paired comparisons; optional stopping
        and selecting a reference after seeing results are not corrected here.
    """
    names = [o.name for o in observables]
    if (
        not names
        or len(set(names)) != len(names)
        or set(names) - set(results.observable_names)
    ):
        msg = "Decision objectives must be distinct names in the results"
        raise ValueError(msg)
    if results.scores.shape != (len(results.configs), len(results.observable_names)):
        msg = "Decision score matrix has an incompatible row/column shape"
        raise ValueError(msg)
    if (
        not 0 < policy.paired_confidence < 1
        or policy.paired_method not in {"bootstrap", "t"}
        or policy.n_boot < 1
    ):
        msg = "Invalid paired inference assumptions"
        raise ValueError(msg)
    policy = deepcopy(policy)
    summary, eligible = _decision_table(results, names, constraints or [])
    raw = np.asarray(summary.scores, dtype=np.float64)
    weights = _weights(policy, names)
    normalized, bounds = _normalize(raw, eligible, names, policy)
    utilities = _utilities(normalized, eligible, weights, observables)
    ranks, selection = _ranks(utilities, eligible)
    pareto, equivalent = _alternatives(raw, eligible, observables, policy)
    report = PreferenceSweep(
        summary,
        weights,
        utilities,
        ranks,
        eligible,
        pareto,
        equivalent,
        bounds,
        {}
        if paired_reference is None
        else _paired(results, summary, eligible, observables, paired_reference, policy),
        policy,
    )
    _decorate_summary(report, selection)
    return report
