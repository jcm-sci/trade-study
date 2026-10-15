"""Context grid for the synthetic serosurvey demonstration (#188).

The serosurvey example fixes one hypothetical population. Whether oversampling
or more communities is the better use of a budget depends on context values that
are not design choices, such as clustering within communities and the antibody
gap between groups. This script runs the 18 designs in each of nine contexts,
keeps context and design factors separate in every configuration, reports the
recommendations within each context, and saves the per-design summaries as a
validated version-1 result bundle with model, context, seed and objective
provenance. All populations, prices and preferences are hypothetical.

Run (regenerates the committed bundle and documentation figure):
    uv run --extra notebook python -m examples.serosurvey_contexts --overwrite
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from dataclasses import dataclass, field
from itertools import pairwise, product
from operator import itemgetter
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

from examples.serosurvey_study import (
    BUDGET,
    COLORS,
    EQUIVALENCE,
    INDICATOR,
    PRECISION_TARGET_PP,
    PRIORITIES,
    REFINE_REPS,
    TARGET_HALF_WIDTH,
    SurveyScorer,
    SurveyWorld,
    acceptable_sets,
    planned_replicates,
    preference_policy,
    robustness_rows,
    survey_factors,
    survey_indicators,
    survey_observables,
)
from trade_study import (
    Constraint,
    Factor,
    FactorType,
    ResultsTable,
    build_grid,
    preference_sweep,
    run_grid,
    save_result_bundle,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.figure import Figure

    from trade_study import Observable, PreferenceSweep, Simulator

BUNDLE_DIR = Path(__file__).resolve().parent / "data" / "serosurvey_contexts"
CONTEXT_SEED = 2027
CORRELATIONS = (0.02, 0.06, 0.12)
UNDERSERVED_PREVALENCES = (0.55, 0.65, 0.75)
BASELINE = {"correlation": 0.06, "underserved_prevalence": 0.65}
DESIGN_KEYS = ("participants", "communities", "allocation")
CONTEXT_KEYS = ("correlation", "underserved_prevalence")


def context_factors() -> list[Factor]:
    """Return the context factors, which describe the setting, not the design.

    Returns:
        Intracluster correlation and the underserved group's prevalence.
    """
    return [
        Factor("correlation", FactorType.DISCRETE, levels=list(CORRELATIONS)),
        Factor(
            "underserved_prevalence",
            FactorType.DISCRETE,
            levels=list(UNDERSERVED_PREVALENCES),
        ),
    ]


def context_grid() -> list[dict[str, Any]]:
    """Cross every context with every design.

    Returns:
        162 configurations, each with separate context and design keys.
    """
    contexts = build_grid(context_factors(), method="full")
    designs = build_grid(survey_factors(), method="full")
    return [{**context, **design} for context, design in product(contexts, designs)]


def split_config(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Separate a configuration into its context and design parts.

    Args:
        config: A context-grid configuration.

    Returns:
        The context values and the design values.

    Raises:
        KeyError: If a context or design key is missing.
    """
    missing = [key for key in (*CONTEXT_KEYS, *DESIGN_KEYS) if key not in config]
    if missing:
        msg = f"Configuration lacks {missing}"
        raise KeyError(msg)
    return (
        {key: config[key] for key in CONTEXT_KEYS},
        {key: config[key] for key in DESIGN_KEYS},
    )


@dataclass(frozen=True)
class ContextWorld:
    """Simulate a design within the context named by its configuration.

    The other group's prevalence stays at the base value. Random streams depend
    on the seed, replicate and design (as in SurveyWorld), not on the context,
    so a design sees the same streams in every context.

    Attributes:
        base: Population whose other-group prevalence and seed are kept.
    """

    base: SurveyWorld = field(default_factory=lambda: SurveyWorld(seed=CONTEXT_SEED))

    def world(self, context: dict[str, Any]) -> SurveyWorld:
        """Return the population for one context.

        Args:
            context: Correlation and underserved prevalence.

        Returns:
            A SurveyWorld with the context's values and the base seed.
        """
        return SurveyWorld(
            prevalence=(
                self.base.prevalence[0],
                float(context["underserved_prevalence"]),
            ),
            correlation=float(context["correlation"]),
            seed=self.base.seed,
        )

    def generate(self, config: dict[str, Any], *, rep: int = 0) -> tuple[Any, Any]:
        """Simulate one survey of the configuration's design in its context.

        Args:
            config: Context and design values.
            rep: Nonnegative replicate id.

        Returns:
            Target group means and one survey's group estimates.
        """
        context, design = split_config(config)
        return self.world(context).generate(design, rep=rep)


@dataclass(frozen=True)
class ContextScorer:
    """Score the design part of a configuration with the example's scorer.

    Attributes:
        scorer: The serosurvey scorer (cost, errors, precision indicator).
    """

    scorer: SurveyScorer = field(default_factory=SurveyScorer)

    def score(
        self, truth: Any, observations: Any, config: dict[str, Any]
    ) -> dict[str, float]:
        """Score one survey; costs depend on the design only.

        Args:
            truth: Target group means.
            observations: Simulated group estimates and field-work counts.
            config: Context and design values.

        Returns:
            Cost, errors and the precision-target indicator.
        """
        return self.scorer.score(truth, observations, split_config(config)[1])


def run_contexts(n_reps: int = REFINE_REPS) -> ResultsTable:
    """Run every design in every context and summarize each pair.

    Args:
        n_reps: Simulated surveys per design and context, fixed in advance.

    Returns:
        One row per context and design, with mean costs and errors, Monte
        Carlo spread and the indicator's fixed-sample Wilson interval.

    Raises:
        ValueError: If n_reps is below the planned indicator precision.
    """
    if n_reps < planned_replicates():
        msg = "n_reps is below the planned indicator precision"
        raise ValueError(msg)
    world: Simulator = ContextWorld()
    raw = run_grid(
        world,
        ContextScorer(),
        context_grid(),
        survey_observables() + survey_indicators(),
        n_reps=n_reps,
    )
    return raw.aggregate_replicates(proportion_observables=[INDICATOR])


def bundle_provenance(n_reps: int) -> dict[str, object]:
    """Describe the model, contexts, seeds and objectives of the bundle.

    Args:
        n_reps: Simulated surveys per design and context.

    Returns:
        Primitive JSON provenance for save_result_bundle.
    """
    here = Path(__file__).resolve().parent
    sources = {
        name: hashlib.sha256((here / name).read_bytes()).hexdigest()
        for name in ("serosurvey_study.py", "serosurvey_contexts.py")
    }
    return {
        "model": "examples/serosurvey_study.py SurveyWorld and SurveyScorer",
        "script": "examples/serosurvey_contexts.py",
        "source_sha256": sources,
        "contexts": {
            "correlation": list(CORRELATIONS),
            "underserved_prevalence": list(UNDERSERVED_PREVALENCES),
            "other_prevalence": float(SurveyWorld().prevalence[0]),
        },
        "designs": list(DESIGN_KEYS),
        "seed": CONTEXT_SEED,
        "random_streams": "seed, replicate and design; shared across contexts",
        "replicates_per_design": n_reps,
        "indicator": {
            "name": INDICATOR,
            "target_pp": PRECISION_TARGET_PP,
            "planned_half_width": TARGET_HALF_WIDTH,
            "interval": "fixed-sample marginal Wilson, 95%",
        },
        "decision": {
            "budget_usd": BUDGET,
            "priorities": {name: dict(w) for name, w in PRIORITIES.items()},
            "equivalence": dict(EQUIVALENCE),
        },
        "hypothetical": True,
    }


def save_contexts(
    summary: ResultsTable, path: Path, *, n_reps: int, overwrite: bool = False
) -> None:
    """Save the per-design summaries as a validated version-1 result bundle.

    Args:
        summary: Output of run_contexts.
        path: Destination directory.
        n_reps: Simulated surveys per design and context.
        overwrite: Remove an existing destination first.
    """
    if overwrite and path.exists():
        shutil.rmtree(path)
    observables: Sequence[Observable] = survey_observables() + survey_indicators()
    save_result_bundle(
        summary,
        path,
        observables=list(observables),
        provenance=bundle_provenance(n_reps),
    )


def context_results(summary: ResultsTable, context: dict[str, Any]) -> ResultsTable:
    """Select one context's per-design rows, keeping design keys only.

    Args:
        summary: Per-design summaries across contexts.
        context: Correlation and underserved prevalence to select.

    Returns:
        The context's rows, with configurations reduced to design keys.
    """
    rows = [
        i
        for i, config in enumerate(summary.configs)
        if split_config(config)[0] == context
    ]
    return ResultsTable(
        configs=[split_config(summary.configs[i])[1] for i in rows],
        scores=summary.scores[rows],
        observable_names=list(summary.observable_names),
        metadata=[dict(summary.metadata[i]) for i in rows],
    )


def context_report(summary: ResultsTable, context: dict[str, Any]) -> PreferenceSweep:
    """Run the example's preference sweep within one context.

    Args:
        summary: Per-design summaries across contexts.
        context: Correlation and underserved prevalence to select.

    Returns:
        The preference report under the example's budget and priorities.
    """
    return preference_sweep(
        context_results(summary, context),
        survey_observables(),
        policy=preference_policy(),
        constraints=[Constraint("financial_budget", "cost_usd", "<=", BUDGET)],
    )


def _label(config: dict[str, Any]) -> str:
    strategy = "O" if config["allocation"] == "oversample" else "P"
    return f"{config['participants']}/{config['communities']}/{strategy}"


def recommendations(summary: ResultsTable) -> list[dict[str, Any]]:
    """Report each context's winners, acceptable sets and steadiest design.

    Args:
        summary: Per-design summaries across contexts.

    Returns:
        One row per context and priority scenario: the winning design
        (participants/communities/allocation), the practically equivalent
        designs, the design with the smallest largest regret, and the winner's
        precision-target share with its interval.
    """
    rows: list[dict[str, Any]] = []
    for correlation, prevalence in product(CORRELATIONS, UNDERSERVED_PREVALENCES):
        context = {"correlation": correlation, "underserved_prevalence": prevalence}
        report = context_report(summary, context)
        configs = report.summary.configs
        sets = acceptable_sets(report)
        robust = robustness_rows(report)
        steadiest = (
            min(robust, key=itemgetter("max_regret"))["design"] if robust else ""
        )
        steadiest_label = (
            _label(configs[int(steadiest[1:]) - 1]) if steadiest else "none"
        )
        for name, ranks in zip(PRIORITIES, report.ranks, strict=True):
            winners = np.flatnonzero(ranks == 1)
            record: dict[str, Any] = {
                **context,
                "priority": name,
                "winner": ", ".join(_label(configs[i]) for i in winners) or "none",
                "allocation": configs[winners[0]]["allocation"] if len(winners) else "",
                "equivalent": ", ".join(
                    _label(configs[i]) for i in sets[name] if i not in winners
                )
                or "none",
                "steadiest": steadiest_label,
            }
            if len(winners):
                interval = report.summary.metadata[winners[0]]["proportion_intervals"][
                    INDICATOR
                ]
                record.update({
                    "within_target": float(interval["estimate"]),
                    "low": float(interval["low"]),
                    "high": float(interval["high"]),
                })
            rows.append(record)
    return rows


def sensitivity(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Count how often the winner changes along each context factor.

    For each priority, compare winners between adjacent levels of one factor
    while the other factor is held fixed.

    Args:
        rows: Output of recommendations.

    Returns:
        By priority, the number of adjacent changes along each context factor
        (six comparisons per factor).
    """
    winners = {
        (row["priority"], row["correlation"], row["underserved_prevalence"]): row[
            "winner"
        ]
        for row in rows
    }
    counts: dict[str, dict[str, int]] = {}
    for name in PRIORITIES:
        along_correlation = sum(
            winners[name, a, p] != winners[name, b, p]
            for p in UNDERSERVED_PREVALENCES
            for a, b in pairwise(CORRELATIONS)
        )
        along_prevalence = sum(
            winners[name, c, a] != winners[name, c, b]
            for c in CORRELATIONS
            for a, b in pairwise(UNDERSERVED_PREVALENCES)
        )
        counts[name] = {
            "correlation": int(along_correlation),
            "underserved_prevalence": int(along_prevalence),
        }
    return counts


def plot_contexts(rows: Sequence[dict[str, Any]]) -> Figure:
    """Show the winning design in each context, one panel per priority.

    Args:
        rows: Output of recommendations.

    Returns:
        Three 3-by-3 panels; outlined cells are the example's baseline context.
    """
    fig, axes = plt.subplots(
        1, len(PRIORITIES), figsize=(14, 4.8), layout="constrained", sharey=True
    )
    lookup = {
        (row["priority"], row["correlation"], row["underserved_prevalence"]): row
        for row in rows
    }
    for ax, name in zip(axes, PRIORITIES, strict=True):
        ax.set_xlim(-0.5, len(UNDERSERVED_PREVALENCES) - 0.5)
        ax.set_ylim(len(CORRELATIONS) - 0.5, -0.5)
        for (r, correlation), (c, prevalence) in product(
            enumerate(CORRELATIONS), enumerate(UNDERSERVED_PREVALENCES)
        ):
            row = lookup[name, correlation, prevalence]
            color = COLORS.get(row["allocation"], "#cccccc")
            ax.add_patch(
                plt.Rectangle(
                    (c - 0.5, r - 0.5),
                    1,
                    1,
                    facecolor=color,
                    alpha=0.85,
                    edgecolor="white",
                    linewidth=1,
                )
            )
            ax.text(
                c,
                r,
                row["winner"],
                ha="center",
                va="center",
                color="white",
                fontsize=11,
            )
        ax.add_patch(
            plt.Rectangle(
                (
                    UNDERSERVED_PREVALENCES.index(BASELINE["underserved_prevalence"])
                    - 0.5,
                    CORRELATIONS.index(BASELINE["correlation"]) - 0.5,
                ),
                1,
                1,
                fill=False,
                edgecolor="black",
                linewidth=3,
                zorder=5,
            )
        )
        ax.set_xticks(
            range(len(UNDERSERVED_PREVALENCES)),
            [f"{100 * p:.0f}%" for p in UNDERSERVED_PREVALENCES],
        )
        ax.set_yticks(range(len(CORRELATIONS)), [f"{c:.2f}" for c in CORRELATIONS])
        ax.set(title=name, xlabel="Underserved-group prevalence")
    axes[0].set_ylabel("Intracluster correlation")
    fig.suptitle(
        "Does the context change the answer? Winning design: participants / "
        "communities / allocation (P proportional, O oversample)\n"
        "Outlined: the example's baseline context; budget and priorities as before",
        fontsize=13,
    )
    return fig


def main(argv: Sequence[str] | None = None) -> None:
    """Run the context grid, save the bundle and the documentation figure.

    Args:
        argv: Command-line arguments (default: sys.argv).
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=BUNDLE_DIR)
    parser.add_argument("--reps", type=int, default=REFINE_REPS)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--figure", type=Path, default=None)
    args = parser.parse_args(argv)
    started = perf_counter()
    summary = run_contexts(args.reps)
    save_contexts(summary, args.out, n_reps=args.reps, overwrite=args.overwrite)
    rows = recommendations(summary)
    figure_path = args.figure or (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "assets"
        / "serosurvey_contexts.png"
    )
    figure = plot_contexts(rows)
    figure.savefig(figure_path, dpi=160)
    plt.close(figure)
    print(
        f"{len(summary.configs)} context-design pairs, {args.reps} surveys each, "
        f"{perf_counter() - started:.1f}s; bundle: {args.out}"
    )
    for name, counts in sensitivity(rows).items():
        print(f"{name}: winner changes {counts}")


if __name__ == "__main__":
    main()
