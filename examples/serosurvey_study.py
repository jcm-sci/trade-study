"""Synthetic serosurvey design for a short IVAC trade-study demonstration.

All populations, prevalences and prices are invented. The model compares
estimation of antibody-status prevalence, not clinical protection.
Run: uv run --extra notebook python examples/serosurvey_study.py
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from operator import itemgetter
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from trade_study import (
    Constraint,
    Direction,
    Factor,
    FactorType,
    Observable,
    Phase,
    PreferencePolicy,
    Study,
    build_grid,
    preference_sweep,
    replicates_for_half_width,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

    from trade_study import PreferenceSweep, ResultsTable

GROUP_WEIGHTS = (0.8, 0.2)
GROUP_NAMES = ("Other communities", "Underserved communities")
COLORS = {"proportional": "#0072B2", "oversample": "#D55E00"}
PRIORITIES = {
    "Cost first": {
        "cost_usd": 0.95,
        "overall_error_pp": 0.025,
        "underserved_error_pp": 0.025,
    },
    "Overall accuracy": {
        "cost_usd": 0.1,
        "overall_error_pp": 0.8,
        "underserved_error_pp": 0.1,
    },
    "Subgroup accuracy": {
        "cost_usd": 0.1,
        "overall_error_pp": 0.1,
        "underserved_error_pp": 0.8,
    },
}
BUDGET = 40_000.0
# One success per simulated survey: underserved estimate within this many points.
PRECISION_TARGET_PP = 5.0
INDICATOR = "underserved_within_5pp"
# Fixed before running: worst-case 95% Wilson half-width for the indicator share.
TARGET_HALF_WIDTH = 0.035
REFINE_REPS = 1000
# Differences this small are treated as practically equal (illustrative judgments).
EQUIVALENCE = {
    "cost_usd": 2_500.0,
    "overall_error_pp": 0.25,
    "underserved_error_pp": 0.5,
}


def allocation_counts(
    config: dict[str, Any],
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Allocate participants and communities to the two population groups.

    Args:
        config: Participant/community totals and allocation label. Totals
            must be positive multiples of ten, with at least one person
            in each community. Proportional allocation is 80:20; the
            oversampling allocation is 50:50.

    Returns:
        Participant counts, then community counts, ordered by group.

    Raises:
        ValueError: If totals or allocation are unsupported.
    """
    n, k = config["participants"], config["communities"]
    if any(
        not isinstance(value, (int, np.integer)) or isinstance(value, bool)
        for value in (n, k)
    ):
        msg = "Participant and community totals must be integers"
        raise ValueError(msg)
    if n < k or k < 10 or n % 10 or k % 10:
        msg = "Use integer multiples of ten with participants >= communities >= 10"
        raise ValueError(msg)
    shares = {"proportional": GROUP_WEIGHTS, "oversample": (0.5, 0.5)}
    if config["allocation"] not in shares:
        msg = "Allocation must be 'proportional' or 'oversample'"
        raise ValueError(msg)
    fraction = shares[config["allocation"]][1]
    second_n, second_k = round(n * fraction), round(k * fraction)
    return (int(n - second_n), second_n), (int(k - second_k), second_k)


@dataclass(frozen=True)
class SurveyOutcome:
    """Group estimates and exact field-work counts for one simulated survey.

    Attributes:
        prevalence: Estimated antibody-status prevalence in each group.
        participants: Participant counts ordered by group.
        communities: Community counts ordered by group.
    """

    prevalence: NDArray[np.float64]
    participants: tuple[int, int]
    communities: tuple[int, int]


@dataclass(frozen=True)
class SurveyWorld:
    """Independent clustered surveys with known hypothetical group means.

    Attributes:
        prevalence: Target means for other/underserved communities.
        correlation: Within-community intraclass correlation. Communities
            are independent, with Beta-distributed probabilities.
        seed: Phase-specific seed; configuration and replicate identify
            independent, reproducible random streams.
    """

    prevalence: tuple[float, float] = (0.9, 0.65)
    correlation: float = 0.06
    seed: int = 2026

    def __post_init__(self) -> None:
        """Validate the illustrative population and random seed.

        Raises:
            ValueError: If probabilities, correlation or seed are invalid.
        """
        if len(self.prevalence) != 2 or not all(
            np.isfinite(p) and 0 <= p <= 1 for p in self.prevalence
        ):
            msg = "Require two finite probabilities between zero and one"
            raise ValueError(msg)
        if not np.isfinite(self.correlation) or not 0 <= self.correlation < 1:
            msg = "Require finite 0 <= correlation < 1"
            raise ValueError(msg)
        if (
            not isinstance(self.seed, int)
            or isinstance(self.seed, bool)
            or self.seed < 0
        ):
            msg = "Require a nonnegative integer seed"
            raise ValueError(msg)

    def generate(
        self, config: dict[str, Any], *, rep: int = 0
    ) -> tuple[NDArray[np.float64], SurveyOutcome]:
        """Simulate clustered counts, preserving exact participant totals.

        Args:
            config: Survey participant/community totals and allocation.
            rep: Nonnegative replicate id. Configurations do not share
                random streams; this example makes no paired-CRN claim.

        Returns:
            Fixed target group means and one survey's group estimates.

        Raises:
            ValueError: If the replicate id or configuration is invalid.
        """
        if not isinstance(rep, int) or isinstance(rep, bool) or rep < 0:
            msg = "Replicate id must be a nonnegative integer"
            raise ValueError(msg)
        people, communities = allocation_counts(config)
        allocation_id = int(config["allocation"] == "oversample")
        rng = np.random.default_rng(
            np.random.SeedSequence([
                self.seed,
                rep,
                int(config["participants"]),
                int(config["communities"]),
                allocation_id,
            ])
        )
        estimates = []
        for p, n, k in zip(self.prevalence, people, communities, strict=True):
            sizes = np.full(k, n // k, dtype=int)
            sizes[: n % k] += 1
            if self.correlation == 0 or p in {0, 1}:
                probabilities = np.full(k, p)
            else:
                concentration = 1 / self.correlation - 1
                probabilities = rng.beta(p * concentration, (1 - p) * concentration, k)
            positives = rng.binomial(sizes, probabilities)
            estimates.append(float(positives.sum() / n))
        return np.array(self.prevalence), SurveyOutcome(
            np.array(estimates), people, communities
        )


@dataclass(frozen=True)
class SurveyCosts:
    """Invented financial costs in illustrative USD, without price calibration.

    Attributes:
        setup: Fixed survey setup cost.
        per_community: Community visit costs ordered by group.
        per_participant: Participant costs ordered by group.
    """

    setup: float = 3000.0
    per_community: tuple[float, float] = (250.0, 700.0)
    per_participant: tuple[float, float] = (30.0, 40.0)

    def __post_init__(self) -> None:
        """Validate the nonnegative financial cost assumptions.

        Raises:
            ValueError: If cost coefficients are invalid.
        """
        if len(self.per_community) != 2 or len(self.per_participant) != 2:
            msg = "Supply one cost per population group"
            raise ValueError(msg)
        values = (self.setup, *self.per_community, *self.per_participant)
        if not all(np.isfinite(value) and value >= 0 for value in values):
            msg = "Costs must be finite and nonnegative"
            raise ValueError(msg)

    def components(self, config: dict[str, Any]) -> dict[str, float]:
        """Compute additive fixed, community and participant costs.

        Args:
            config: Survey participant/community totals and allocation.

        Returns:
            Three separate financial cost components in illustrative USD.
        """
        people, communities = allocation_counts(config)
        return {
            "Setup": self.setup,
            "Community visits": float(np.dot(communities, self.per_community)),
            "Participants": float(np.dot(people, self.per_participant)),
        }


@dataclass(frozen=True)
class SurveyScorer:
    """Financial cost and absolute prevalence error, with population weighting.

    Attributes:
        costs: Illustrative financial cost coefficients.
    """

    costs: SurveyCosts = field(default_factory=SurveyCosts)

    def score(
        self,
        truth: NDArray[np.float64],
        observations: SurveyOutcome,
        config: dict[str, Any],
    ) -> dict[str, float]:
        """Score one survey; averaging errors gives mean absolute error.

        Args:
            truth: Target antibody-status prevalences, ordered by group.
            observations: Simulated group estimates and field-work counts.
            config: Survey configuration used for financial cost.

        Returns:
            Cost, overall/subgroup absolute errors in percentage points, and a
            0/1 indicator that the subgroup error is at most PRECISION_TARGET_PP
            (inclusive; the small tolerance absorbs floating-point rounding).
        """
        overall = float(np.dot(observations.prevalence - truth, GROUP_WEIGHTS))
        subgroup = 100 * abs(float(observations.prevalence[1] - truth[1]))
        return {
            "cost_usd": sum(self.costs.components(config).values()),
            "overall_error_pp": 100 * abs(overall),
            "underserved_error_pp": subgroup,
            INDICATOR: float(subgroup <= PRECISION_TARGET_PP + 1e-9),
        }


def survey_factors() -> list[Factor]:
    """Return the factors for the 18-design demonstration.

    Returns:
        Participant count, community count and allocation factors.
    """
    return [
        Factor("participants", FactorType.DISCRETE, levels=[300, 600, 1200]),
        Factor("communities", FactorType.DISCRETE, levels=[10, 20, 40]),
        Factor(
            "allocation", FactorType.CATEGORICAL, levels=["proportional", "oversample"]
        ),
    ]


def survey_observables() -> list[Observable]:
    """Return the three objectives, all minimized.

    Returns:
        Financial cost, overall MAE and underserved-group MAE observables.
    """
    return [Observable(name, Direction.MINIMIZE) for name in PRIORITIES["Cost first"]]


def survey_indicators() -> list[Observable]:
    """Return reported 0/1 indicators that do not enter the preference weights.

    Returns:
        The precision-target indicator, maximized when reported.
    """
    return [Observable(INDICATOR, Direction.MAXIMIZE)]


def planned_replicates() -> int:
    """Return the replicate count needed for the planned indicator precision.

    The worst-case 95% Wilson half-width (at an observed share of one half) is
    at most TARGET_HALF_WIDTH with this many independent simulated surveys. The
    count is fixed before running and does not depend on the outcomes.

    Returns:
        Minimum replicate count per design for the planned precision.
    """
    return replicates_for_half_width(TARGET_HALF_WIDTH, confidence=0.95)


def preference_policy() -> PreferencePolicy:
    """Return three hypothetical priorities with fixed reference anchors.

    Returns:
        An explicit policy; anchors are scaling choices, not eligibility limits.
        Practical-equivalence tolerances are raw-unit judgments (EQUIVALENCE).
    """
    return PreferencePolicy(
        weights=list(PRIORITIES.values()),
        normalization="reference",
        reference_bounds={
            "cost_usd": (0, 70_000),
            "overall_error_pp": (0, 4),
            "underserved_error_pp": (0, 10),
        },
        equivalence=dict(EQUIVALENCE),
    )


def plot_population(world: SurveyWorld) -> Figure:
    """Draw the population assumptions, rather than estimated outcomes.

    Args:
        world: Hypothetical group prevalences.

    Returns:
        A two-panel population-share and antibody-status prevalence figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    for ax, values, title in zip(
        axes,
        (GROUP_WEIGHTS, world.prevalence),
        ("Population composition", "Assumed antibody-status prevalence"),
        strict=True,
    ):
        bars = ax.bar(GROUP_NAMES, 100 * np.array(values), color=list(COLORS.values()))
        ax.bar_label(bars, fmt="%.0f%%", padding=4, fontsize=13)
        ax.set(ylim=(0, 110), ylabel="Percent", title=title)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("A fictional population: assumptions, not measured data", fontsize=16)
    return fig


def _selected_indices(report: PreferenceSweep) -> NDArray[np.intp]:
    return np.flatnonzero(np.any(report.ranks == 1, axis=0))


def acceptable_sets(report: PreferenceSweep) -> dict[str, list[int]]:
    """Return the designs practically equivalent to each scenario's winner.

    Two designs are practically equivalent when every objective differs by at
    most the policy's raw-unit tolerance. The relation is pairwise and need not
    be transitive, so each set is anchored on the scenario's winners.

    Args:
        report: Preference report with its equivalence matrix.

    Returns:
        Feasible design indices by priority scenario, winners included; empty
        when no design is feasible.
    """
    sets: dict[str, list[int]] = {}
    for name, ranks in zip(PRIORITIES, report.ranks, strict=True):
        winners = np.flatnonzero(ranks == 1)
        members = np.any(report.equivalent[winners], axis=0)
        sets[name] = np.flatnonzero(members).tolist()
    return sets


def robustness_rows(report: PreferenceSweep) -> list[dict[str, Any]]:
    """Summarize how each feasible Pareto design fares across the priorities.

    Regret is a design's weighted, reference-scaled loss minus the scenario
    winner's loss. The maximum over the supplied scenarios shows how much a
    choice can lose if priorities differ; it describes these scenarios only.

    Args:
        report: Preference report under the named scenarios.

    Returns:
        One row per feasible Pareto design, sorted by cost, with the scenarios
        it wins, its rank range, its largest regret and the scenarios whose
        acceptable set includes it.
    """
    acceptable = acceptable_sets(report)
    indices = np.flatnonzero(report.pareto)
    indices = indices[np.argsort(report.summary.scores[indices, 0], kind="stable")]
    rows: list[dict[str, Any]] = []
    for i in indices:
        decision = report.summary.metadata[i]["decision"]
        cost, overall, subgroup = report.summary.scores[i]
        wins = [
            name
            for name, ranks in zip(PRIORITIES, report.ranks, strict=True)
            if ranks[i] == 1
        ]
        near = [name for name, members in acceptable.items() if i in members]
        rows.append({
            "design": f"D{i + 1:02d}",
            **report.summary.configs[i],
            "cost_usd": float(cost),
            "overall_mae_pp": float(overall),
            "subgroup_mae_pp": float(subgroup),
            "wins": ", ".join(wins) or "none",
            "acceptable_for": ", ".join(near) or "none",
            "rank_range": (
                f"{decision['best_rank']:.0f} to {decision['worst_rank']:.0f}"
            ),
            "max_regret": float(decision["max_regret"]),
        })
    return rows


def precision_rows(
    report: PreferenceSweep, summary: ResultsTable
) -> list[dict[str, Any]]:
    """Report the precision-target share for each feasible Pareto design.

    Args:
        report: Preference report defining feasibility and the Pareto set.
        summary: Per-design results from ``aggregate_replicates`` with
            ``proportion_observables=[INDICATOR]``.

    Returns:
        One row per feasible Pareto design, sorted by cost, with the share of
        simulated surveys whose subgroup estimate is within the target and its
        fixed-sample marginal Wilson interval.
    """
    intervals = {
        tuple(sorted(config.items())): meta["proportion_intervals"][INDICATOR]
        for config, meta in zip(summary.configs, summary.metadata, strict=True)
        if INDICATOR in meta.get("proportion_intervals", {})
    }
    indices = np.flatnonzero(report.pareto)
    indices = indices[np.argsort(report.summary.scores[indices, 0], kind="stable")]
    rows: list[dict[str, Any]] = []
    for i in indices:
        config = report.summary.configs[i]
        record = intervals[tuple(sorted(config.items()))]
        rows.append({
            "design": f"D{i + 1:02d}",
            **config,
            "cost_usd": float(report.summary.scores[i, 0]),
            "subgroup_mae_pp": float(report.summary.scores[i, 2]),
            "within_target": float(record["estimate"]),
            "low": float(record["low"]),
            "high": float(record["high"]),
            "k": int(record["k"]),
            "n": int(record["n"]),
        })
    return rows


def _mc_errors(results: ResultsTable, name: str) -> NDArray[np.float64]:
    return np.array([
        meta["score_std"][name] / np.sqrt(meta["n_reps"] - 1)
        for meta in results.metadata
    ])


def plot_tradeoffs(report: PreferenceSweep, budget: float) -> Figure:
    """Project the three-objective feasible Pareto set into two panels.

    Args:
        report: Preference report, including budget feasibility and Pareto flags.
        budget: Financial ceiling in illustrative USD.

    Returns:
        Cost versus overall/subgroup MAE, with approximate Monte Carlo bars.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), layout="constrained")
    table = report.summary
    costs = table.scores[:, 0] / 1000
    selected = _selected_indices(report)
    for ax, name, title in zip(
        axes,
        ("overall_error_pp", "underserved_error_pp"),
        ("Overall population", "Underserved group"),
        strict=True,
    ):
        errors = table.scores[:, table.observable_names.index(name)]
        ax.errorbar(
            costs,
            errors,
            yerr=2 * _mc_errors(table, name),
            fmt="none",
            ecolor="#bbbbbb",
            capsize=2,
            zorder=1,
        )
        for i, config in enumerate(table.configs):
            ax.scatter(
                costs[i],
                errors[i],
                s=75,
                marker="o" if config["allocation"] == "proportional" else "D",
                color=COLORS[config["allocation"]] if report.feasible[i] else "#cccccc",
                edgecolors="black" if report.pareto[i] else "none",
                linewidths=1.4,
                zorder=3,
            )
        for offset, i in enumerate(selected):
            ax.annotate(
                f"D{i + 1:02d}",
                (costs[i], errors[i]),
                xytext=(5, 10 + 10 * (offset % 2)),
                textcoords="offset points",
                fontsize=10,
                fontweight="bold",
            )
        ax.axvline(budget / 1000, color="#555555", linestyle="--")
        ax.set(
            xlabel="Financial cost (illustrative USD thousands)",
            ylabel="Mean absolute error (percentage points)",
            title=title,
        )
        ax.spines[["top", "right"]].set_visible(False)
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            color=COLORS["proportional"],
            label="Proportional allocation",
        ),
        Line2D(
            [],
            [],
            marker="D",
            linestyle="",
            color=COLORS["oversample"],
            label="Oversample underserved group",
        ),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            color="white",
            markeredgecolor="black",
            label="Feasible Pareto design (all 3 objectives)",
        ),
        Line2D([], [], marker="o", linestyle="", color="#cccccc", label="Over budget"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, fontsize=10)
    fig.suptitle("Cost, overall accuracy and subgroup accuracy compete", fontsize=16)
    return fig


def plot_priorities(report: PreferenceSweep) -> Figure:
    """Show ranks among feasible Pareto candidates under three priorities.

    Args:
        report: Results of the three named preference scenarios.

    Returns:
        A rank heatmap; highlighted cells identify winners, including ties.
    """
    indices = np.flatnonzero(report.pareto)
    indices = indices[np.argsort(report.summary.scores[indices, 0])]
    ranks = report.ranks[:, indices].T
    fig, ax = plt.subplots(
        figsize=(10, max(3.5, 0.42 * len(indices) + 1.4)), layout="constrained"
    )
    if not len(indices):
        ax.text(0.5, 0.5, "No design meets the budget", ha="center", va="center")
        ax.axis("off")
        return fig
    ax.imshow(
        ranks, cmap="Blues_r", vmin=1, vmax=max(2, float(ranks.max())), aspect="auto"
    )
    acceptable = acceptable_sets(report)
    labels = []
    for i in indices:
        cfg = report.summary.configs[i]
        strategy = "P" if cfg["allocation"] == "proportional" else "O"
        labels.append(
            f"D{i + 1:02d} · {cfg['participants']} people / "
            f"{cfg['communities']} communities / {strategy}"
        )
    ax.set_yticks(np.arange(len(indices)), labels, fontsize=10)
    ax.set_xticks(np.arange(len(PRIORITIES)), list(PRIORITIES), fontsize=11)
    for row, column in np.ndindex(ranks.shape):
        rank = ranks[row, column]
        ax.text(
            column,
            row,
            f"{rank:.0f}",
            ha="center",
            va="center",
            color="white" if rank < (ranks.max() + 1) / 2 else "black",
            fontweight="bold" if rank == 1 else "normal",
        )
        if rank != 1 and indices[row] in acceptable[list(PRIORITIES)[column]]:
            ax.add_patch(
                Rectangle(
                    (column - 0.5, row - 0.5),
                    1,
                    1,
                    fill=False,
                    edgecolor="black",
                    linewidth=2.5,
                )
            )
    ax.set_title(
        "Same evidence, different priorities\n"
        "Rank 1 wins; outlined designs are practically equivalent to the winner",
        fontsize=15,
        pad=16,
    )
    ax.set_xlabel(
        "Allocation: P = proportional; O = oversample. "
        "Ranks include all feasible designs.",
        labelpad=12,
    )
    return fig


def plot_robustness(report: PreferenceSweep) -> Figure:
    """Show each feasible Pareto design's largest regret across priorities.

    Args:
        report: Preference report under the named scenarios.

    Returns:
        A horizontal bar chart annotated with the scenarios each design wins
        or nearly wins, or a message if no design is feasible.
    """
    rows = robustness_rows(report)
    fig, ax = plt.subplots(
        figsize=(10, max(3.5, 0.42 * len(rows) + 1.8)), layout="constrained"
    )
    if not rows:
        ax.text(0.5, 0.5, "No design meets the budget", ha="center", va="center")
        ax.axis("off")
        return fig
    regrets = np.array([row["max_regret"] for row in rows])
    lowest = np.isclose(regrets, regrets.min())
    positions = np.arange(len(rows))
    ax.barh(
        positions,
        regrets,
        color=[COLORS[row["allocation"]] for row in rows],
        edgecolor=["black" if flag else "none" for flag in lowest],
        linewidth=2,
    )
    labels = [
        f"{row['design']} · {row['participants']} people / "
        f"{row['communities']} communities"
        for row in rows
    ]
    ax.set_yticks(positions, labels, fontsize=10)
    ax.invert_yaxis()
    pad = 0.01 * regrets.max()
    for position, row in zip(positions, rows, strict=True):
        notes = []
        if row["wins"] != "none":
            notes.append(f"wins: {row['wins']}")
        near = [
            name
            for name in row["acceptable_for"].split(", ")
            if name != "none" and name not in row["wins"].split(", ")
        ]
        if near:
            notes.append(f"near the winner: {', '.join(near)}")
        ax.text(
            row["max_regret"] + pad,
            position,
            "; ".join(notes),
            va="center",
            fontsize=9,
        )
    ax.set_xlim(0, 1.9 * regrets.max())
    ax.set(
        xlabel="Largest regret across the three priorities "
        "(weighted share of the reference ranges)",
        title="Which choices hold up when priorities change?\n"
        "Outlined: the smallest largest regret across these three scenarios",
    )
    ax.legend(
        handles=[
            Line2D([], [], color=color, lw=8, label=name)
            for name, color in COLORS.items()
        ],
        loc="lower right",
        frameon=False,
    )
    ax.spines[["top", "right"]].set_visible(False)
    return fig


def plot_precision(report: PreferenceSweep, summary: ResultsTable) -> Figure:
    """Show how often each feasible Pareto design meets the subgroup target.

    Args:
        report: Preference report defining feasibility and the Pareto set.
        summary: Per-design results with indicator interval metadata.

    Returns:
        Shares with 95% Wilson intervals, or a message if no design is feasible.
    """
    rows = precision_rows(report, summary)
    fig, ax = plt.subplots(
        figsize=(10, max(3.5, 0.42 * len(rows) + 1.8)), layout="constrained"
    )
    if not rows:
        ax.text(0.5, 0.5, "No design meets the budget", ha="center", va="center")
        ax.axis("off")
        return fig
    positions = np.arange(len(rows))
    shares = np.array([row["within_target"] for row in rows])
    errors = np.array([
        [row["within_target"] - row["low"], row["high"] - row["within_target"]]
        for row in rows
    ]).T
    ax.barh(
        positions,
        100 * shares,
        xerr=100 * errors,
        color=[COLORS[row["allocation"]] for row in rows],
        error_kw={"ecolor": "black", "capsize": 3},
    )
    labels = [
        f"{row['design']} · {row['participants']} people / "
        f"{row['communities']} communities"
        for row in rows
    ]
    ax.set_yticks(positions, labels, fontsize=10)
    ax.invert_yaxis()
    for position, row in zip(positions, rows, strict=True):
        ax.text(
            100 * row["high"] + 1,
            position,
            f"{100 * row['within_target']:.0f}% "
            f"({100 * row['low']:.0f} to {100 * row['high']:.0f})",
            va="center",
            fontsize=9,
        )
    ax.set_xlim(0, 115)
    n = rows[0]["n"]
    ax.set(
        xlabel=f"Simulated surveys with the subgroup estimate within "
        f"±{PRECISION_TARGET_PP:.0f} points (percent of {n:,})",
        title="How often is the subgroup estimate close enough?\n"
        "Bars: 95% Wilson intervals for simulation precision under this model",
    )
    ax.legend(
        handles=[
            Line2D([], [], color=color, lw=8, label=name)
            for name, color in COLORS.items()
        ],
        loc="lower right",
        frameon=False,
    )
    ax.spines[["top", "right"]].set_visible(False)
    return fig


def plot_cost_components(report: PreferenceSweep, costs: SurveyCosts) -> Figure:
    """Show financial cost components for the preference winners.

    Args:
        report: Preference report identifying selected candidates.
        costs: Illustrative financial cost model.

    Returns:
        A stacked cost figure, or a message if no design is feasible.
    """
    indices = _selected_indices(report)
    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    if not len(indices):
        ax.text(0.5, 0.5, "No design meets the budget", ha="center", va="center")
        ax.axis("off")
        return fig
    components = [costs.components(report.summary.configs[i]) for i in indices]
    bottom = np.zeros(len(indices))
    for name, color in zip(
        components[0], ("#666666", "#009E73", "#E69F00"), strict=True
    ):
        values = np.array([row[name] for row in components]) / 1000
        ax.bar(
            [f"D{i + 1:02d}" for i in indices],
            values,
            bottom=bottom,
            label=name,
            color=color,
        )
        bottom += values
    ax.set(
        ylabel="Financial cost (illustrative USD thousands)",
        title="Why do the selected designs cost different amounts?",
    )
    ax.legend(loc="upper left", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    return fig


def run_study(
    world: SurveyWorld,
) -> tuple[ResultsTable, ResultsTable, PreferenceSweep]:
    """Run the screening and refinement phases and the budgeted preference sweep.

    Args:
        world: Hypothetical population for the screening phase; refinement uses
            an independent seed.

    Returns:
        Raw refinement results, per-design results with indicator intervals,
        and the preference report under the illustrative budget.

    Raises:
        ValueError: If REFINE_REPS is below the planned indicator precision.
    """
    if planned_replicates() > REFINE_REPS:
        msg = "REFINE_REPS is below the planned indicator precision"
        raise ValueError(msg)
    factors = survey_factors()
    observables = survey_observables()
    grid = build_grid(factors, method="full")
    study = Study(
        world=world,
        scorer=SurveyScorer(),
        observables=observables + survey_indicators(),
        factors=factors,
        phases=[
            Phase("screen", grid=grid, n_reps=100),
            Phase(
                "refine",
                grid=grid,
                n_reps=REFINE_REPS,
                world=replace(world, seed=2027),
            ),
        ],
    )
    study.run()
    raw = study.results("refine")
    summary = raw.aggregate_replicates(proportion_observables=[INDICATOR])
    report = preference_sweep(
        raw,
        observables,
        policy=preference_policy(),
        constraints=[Constraint("financial_budget", "cost_usd", "<=", BUDGET)],
    )
    return raw, summary, report


def main() -> None:
    """Run the notebook's study and export documentation figures.

    All 18 designs receive both screening and independent refinement.
    """
    started = perf_counter()
    world = SurveyWorld(seed=2026)
    _, summary, report = run_study(world)
    asset_dir = Path(__file__).resolve().parents[1] / "docs" / "assets"
    for name, figure in {
        "population": plot_population(world),
        "tradeoffs": plot_tradeoffs(report, BUDGET),
        "priorities": plot_priorities(report),
        "robustness": plot_robustness(report),
        "precision": plot_precision(report, summary),
        "costs": plot_cost_components(report, SurveyCosts()),
    }.items():
        figure.savefig(asset_dir / f"serosurvey_{name}.png", dpi=160)
        plt.close(figure)
    print(
        f"18 designs, 19,800 evaluations, completed in {perf_counter() - started:.2f}s"
    )
    acceptable = acceptable_sets(report)
    for name, ranks in zip(PRIORITIES, report.ranks, strict=True):
        selected = np.flatnonzero(ranks == 1)
        near = ", ".join(f"D{i + 1:02d}" for i in acceptable[name])
        print(
            f"{name}: "
            + ", ".join(f"D{i + 1:02d}" for i in selected)
            + f" (practically equivalent: {near})"
        )
    for row in precision_rows(report, summary):
        print(
            f"{row['design']}: {100 * row['within_target']:.1f}% within "
            f"±{PRECISION_TARGET_PP:.0f} points "
            f"({100 * row['low']:.1f} to {100 * row['high']:.1f}; n={row['n']})"
        )
    rows = robustness_rows(report)
    steadiest = min(rows, key=itemgetter("max_regret"))
    print(
        f"Smallest largest regret: {steadiest['design']} "
        f"({steadiest['max_regret']:.3f}; wins: {steadiest['wins']})"
    )


if __name__ == "__main__":
    main()
