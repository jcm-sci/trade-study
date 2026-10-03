"""Synthetic serosurvey design for a short IVAC trade-study demonstration.

All populations, prevalences and prices are invented. The model compares
estimation of antibody-status prevalence, not clinical protection.
Run: uv run --extra notebook python examples/serosurvey_study.py
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

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
            Cost and overall/subgroup absolute errors in percentage points.
        """
        overall = float(np.dot(observations.prevalence - truth, GROUP_WEIGHTS))
        return {
            "cost_usd": sum(self.costs.components(config).values()),
            "overall_error_pp": 100 * abs(overall),
            "underserved_error_pp": 100 * abs(observations.prevalence[1] - truth[1]),
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


def preference_policy() -> PreferencePolicy:
    """Return three hypothetical priorities with fixed reference anchors.

    Returns:
        An explicit policy; anchors are scaling choices, not eligibility limits.
    """
    return PreferencePolicy(
        weights=list(PRIORITIES.values()),
        normalization="reference",
        reference_bounds={
            "cost_usd": (0, 70_000),
            "overall_error_pp": (0, 4),
            "underserved_error_pp": (0, 10),
        },
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
    ax.set_title(
        "Same evidence, different priorities\n"
        "Rank 1 wins; ranks include all feasible designs",
        fontsize=15,
        pad=16,
    )
    ax.set_xlabel("Allocation: P = proportional; O = oversample", labelpad=12)
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


def main() -> None:
    """Run the notebook's study and export documentation figures.

    All 18 designs receive both screening and independent refinement.
    """
    started = perf_counter()
    factors = survey_factors()
    observables = survey_observables()
    grid = build_grid(factors, method="full")
    world = SurveyWorld(seed=2026)
    study = Study(
        world=world,
        scorer=SurveyScorer(),
        observables=observables,
        factors=factors,
        phases=[
            Phase("screen", grid=grid, n_reps=100),
            Phase("refine", grid=grid, n_reps=1000, world=replace(world, seed=2027)),
        ],
    )
    study.run()
    report = preference_sweep(
        study.results("refine"),
        observables,
        policy=preference_policy(),
        constraints=[Constraint("financial_budget", "cost_usd", "<=", BUDGET)],
    )
    asset_dir = Path(__file__).resolve().parents[1] / "docs" / "assets"
    for name, figure in {
        "population": plot_population(world),
        "tradeoffs": plot_tradeoffs(report, BUDGET),
        "priorities": plot_priorities(report),
        "costs": plot_cost_components(report, SurveyCosts()),
    }.items():
        figure.savefig(asset_dir / f"serosurvey_{name}.png", dpi=160)
        plt.close(figure)
    print(
        f"18 designs, 19,800 evaluations, completed in {perf_counter() - started:.2f}s"
    )
    for name, ranks in zip(PRIORITIES, report.ranks, strict=True):
        selected = np.flatnonzero(ranks == 1)
        print(f"{name}: " + ", ".join(f"D{i + 1:02d}" for i in selected))


if __name__ == "__main__":
    main()
