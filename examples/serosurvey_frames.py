"""Sampling-frame comparison for the synthetic serosurvey demonstration (#191).

Residual blood specimens collected at health facilities can cost less than a
community survey, but they may over- or under-represent antibody-positive
people, for example when facility use is linked to vaccination. This script
adds a sampling-frame factor (community survey or residual specimens) to the
18 designs, runs the 36 designs under four values of a selection-bias context,
reports the recommendations within each context and the smallest subgroup
error each frame reaches within the budget, and saves the per-design summaries
as a validated version-1 result bundle. All populations, prices, bias values
and preferences are hypothetical.

Run (regenerates the committed bundle and documentation figure):
    uv run --extra notebook python -m examples.serosurvey_frames --overwrite
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from dataclasses import asdict, dataclass, field, replace
from itertools import product
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

from examples.serosurvey_study import (
    BUDGET,
    EQUIVALENCE,
    FRAMES,
    INDICATOR,
    PRECISION_TARGET_PP,
    PRIORITIES,
    REFINE_REPS,
    SurveyCosts,
    SurveyScorer,
    SurveyWorld,
    acceptable_sets,
    planned_replicates,
    preference_policy,
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

BUNDLE_DIR = Path(__file__).resolve().parent / "data" / "serosurvey_frames"
FRAME_SEED = 2027
SELECTION_ODDS = (1.0, 1.1, 1.25, 1.5)
DESIGN_KEYS = ("participants", "communities", "allocation", "frame")
CONTEXT_KEYS = ("selection_odds",)
FRAME_COLORS = {"community": "#009E73", "residual": "#CC79A7"}
BIAS_REPS = 200


def frame_factors() -> list[Factor]:
    """Return the design factors with the sampling frame added.

    Returns:
        Participant count, community count, allocation and frame factors.
    """
    return [
        *survey_factors(),
        Factor("frame", FactorType.CATEGORICAL, levels=list(FRAMES)),
    ]


def frame_grid() -> list[dict[str, Any]]:
    """Cross every selection-bias context with every design.

    Returns:
        144 configurations, each with separate context and design keys.
    """
    designs = build_grid(frame_factors(), method="full")
    return [
        {"selection_odds": odds, **design}
        for odds, design in product(SELECTION_ODDS, designs)
    ]


def split_config(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Separate a configuration into its context and design parts.

    Args:
        config: A frame-grid configuration.

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
class FrameWorld:
    """Simulate a design under the selection bias named by its configuration.

    The selection odds apply to both groups. Community designs do not depend on
    them, and every design uses the base population and draws of each
    replicate, so residual and community designs are compared under common
    random numbers.

    Attributes:
        base: Population whose other values and seed are kept.
    """

    base: SurveyWorld = field(default_factory=lambda: SurveyWorld(seed=FRAME_SEED))

    def world(self, context: dict[str, Any]) -> SurveyWorld:
        """Return the population for one context.

        Args:
            context: Selection odds for residual specimens.

        Returns:
            The base SurveyWorld with the context's selection odds in both groups.
        """
        odds = float(context["selection_odds"])
        return replace(self.base, selection_odds=(odds, odds))

    def generate(self, config: dict[str, Any], *, rep: int = 0) -> tuple[Any, Any]:
        """Simulate one survey of the configuration's design in its context.

        Args:
            config: Context and design values.
            rep: Nonnegative replicate id.

        Returns:
            Target group prevalences and one survey's group estimates.
        """
        context, design = split_config(config)
        return self.world(context).generate(design, rep=rep)


@dataclass(frozen=True)
class FrameScorer:
    """Score the design part of a configuration with the example's scorer.

    Attributes:
        scorer: The serosurvey scorer; costs depend on the frame.
    """

    scorer: SurveyScorer = field(default_factory=SurveyScorer)

    def score(
        self, truth: Any, observations: Any, config: dict[str, Any]
    ) -> dict[str, float]:
        """Score one survey; costs depend on the design only.

        Args:
            truth: Target group prevalences.
            observations: Simulated group estimates and field-work counts.
            config: Context and design values.

        Returns:
            Cost, errors and the precision-target indicator.
        """
        return self.scorer.score(truth, observations, split_config(config)[1])


def run_frames(n_reps: int = REFINE_REPS) -> ResultsTable:
    """Run every design in every selection-bias context and summarize each pair.

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
    world: Simulator = FrameWorld()
    raw = run_grid(
        world,
        FrameScorer(),
        frame_grid(),
        survey_observables() + survey_indicators(),
        n_reps=n_reps,
    )
    return raw.aggregate_replicates(proportion_observables=[INDICATOR])


def selection_bias_pp(odds: float, reps: int = BIAS_REPS) -> tuple[float, float]:
    """Average shift of residual-specimen prevalence from the population.

    Args:
        odds: Selection odds in both groups.
        reps: Replicate populations averaged.

    Returns:
        The expected residual-specimen prevalence minus the realized population
        prevalence, in percentage points, for each group.
    """
    base = FrameWorld().base
    shifts = []
    for rep in range(reps):
        prevalences, _ = base.population(rep)
        shifts.append([
            100 * float(np.mean(odds * pi / (odds * pi + 1 - pi) - pi))
            for pi in prevalences
        ])
    other, underserved = np.mean(shifts, axis=0)
    return float(other), float(underserved)


def bundle_provenance(n_reps: int) -> dict[str, object]:
    """Describe the model, contexts, costs, seeds and objectives of the bundle.

    Args:
        n_reps: Simulated surveys per design and context.

    Returns:
        Primitive JSON provenance for save_result_bundle.
    """
    base = FrameWorld().base
    here = Path(__file__).resolve().parent
    sources = {
        name: hashlib.sha256((here / name).read_bytes()).hexdigest()
        for name in ("serosurvey_study.py", "serosurvey_frames.py")
    }
    return {
        "model": "examples/serosurvey_study.py SurveyWorld and SurveyScorer",
        "script": "examples/serosurvey_frames.py",
        "source_sha256": sources,
        "contexts": {
            "selection_odds": list(SELECTION_ODDS),
            "applies_to": "both groups",
            "prevalence": list(base.prevalence),
            "correlation": base.correlation,
        },
        "selection_model": (
            "residual specimens from a community's facility: odds of antibody"
            " positivity = selection_odds x community odds"
        ),
        "costs": {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in asdict(SurveyCosts()).items()
        },
        "designs": list(DESIGN_KEYS),
        "seed": FRAME_SEED,
        "random_streams": (
            "seed and replicate; one population per replicate shared by all"
            " designs and frames"
        ),
        "population": {
            "estimand": base.estimand,
            "communities_per_group": list(base.pool),
            "sampled_cap": base.sampled_cap,
            "per_community_cap": base.per_community_cap,
        },
        "replicates_per_design": n_reps,
        "indicator": {
            "name": INDICATOR,
            "target_pp": PRECISION_TARGET_PP,
            "interval": "fixed-sample marginal Wilson, 95%",
        },
        "decision": {
            "budget_usd": BUDGET,
            "priorities": {name: dict(w) for name, w in PRIORITIES.items()},
            "equivalence": dict(EQUIVALENCE),
        },
        "hypothetical": True,
    }


def save_frames(
    summary: ResultsTable, path: Path, *, n_reps: int, overwrite: bool = False
) -> None:
    """Save the per-design summaries as a validated version-1 result bundle.

    Args:
        summary: Output of run_frames.
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


def frame_results(summary: ResultsTable, context: dict[str, Any]) -> ResultsTable:
    """Select one context's per-design rows, keeping design keys only.

    Args:
        summary: Per-design summaries across contexts.
        context: Selection odds to select.

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


def frame_report(summary: ResultsTable, context: dict[str, Any]) -> PreferenceSweep:
    """Run the example's preference sweep within one context.

    Args:
        summary: Per-design summaries across contexts.
        context: Selection odds to select.

    Returns:
        The preference report under the example's budget and priorities.
    """
    return preference_sweep(
        frame_results(summary, context),
        survey_observables(),
        policy=preference_policy(),
        constraints=[Constraint("financial_budget", "cost_usd", "<=", BUDGET)],
    )


def design_label(config: dict[str, Any]) -> str:
    """Return a compact design label.

    Args:
        config: Design values.

    Returns:
        Participants/communities/allocation/frame, for example ``600/20/O/R``
        (P proportional, O oversample; C community survey, R residual specimens).
    """
    strategy = "O" if config["allocation"] == "oversample" else "P"
    frame = "R" if config["frame"] == "residual" else "C"
    return f"{config['participants']}/{config['communities']}/{strategy}/{frame}"


def recommendations(summary: ResultsTable) -> list[dict[str, Any]]:
    """Report each context's winners and practically equivalent designs.

    Args:
        summary: Per-design summaries across contexts.

    Returns:
        One row per context and priority scenario: the winning design and its
        frame, the practically equivalent designs, and the winner's
        precision-target share with its interval.
    """
    rows: list[dict[str, Any]] = []
    for odds in SELECTION_ODDS:
        context = {"selection_odds": odds}
        report = frame_report(summary, context)
        configs = report.summary.configs
        sets = acceptable_sets(report)
        for name, ranks in zip(PRIORITIES, report.ranks, strict=True):
            winners = np.flatnonzero(ranks == 1)
            record: dict[str, Any] = {
                **context,
                "priority": name,
                "winner": ", ".join(design_label(configs[i]) for i in winners)
                or "none",
                "frame": configs[winners[0]]["frame"] if len(winners) else "",
                "equivalent": ", ".join(
                    design_label(configs[i]) for i in sets[name] if i not in winners
                )
                or "none",
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


def best_by_frame(summary: ResultsTable) -> list[dict[str, Any]]:
    """Find the smallest subgroup error each frame reaches within the budget.

    Args:
        summary: Per-design summaries across contexts.

    Returns:
        One row per context and frame: the design with the smallest subgroup
        MAE among those within the budget, its cost, MAE and Monte Carlo
        standard error, and the frame's expected subgroup selection bias.
    """
    cost = summary.observable_names.index("cost_usd")
    error = summary.observable_names.index("underserved_error_pp")
    rows: list[dict[str, Any]] = []
    for odds in SELECTION_ODDS:
        table = frame_results(summary, {"selection_odds": odds})
        bias = selection_bias_pp(odds)[1]
        for frame in FRAMES:
            candidates = [
                i
                for i, config in enumerate(table.configs)
                if config["frame"] == frame and table.scores[i, cost] <= BUDGET
            ]
            if not candidates:
                continue
            best = candidates[int(np.argmin(table.scores[candidates, error]))]
            meta = table.metadata[best]
            rows.append({
                "selection_odds": odds,
                "frame": frame,
                "design": design_label(table.configs[best]),
                "cost_usd": float(table.scores[best, cost]),
                "subgroup_mae_pp": float(table.scores[best, error]),
                "mcse_pp": float(
                    meta["score_std"]["underserved_error_pp"]
                    / np.sqrt(meta["n_reps"] - 1)
                ),
                "bias_pp": bias if frame == "residual" else 0.0,
            })
    return rows


def plot_frames(
    best: Sequence[dict[str, Any]], rows: Sequence[dict[str, Any]]
) -> Figure:
    """Show where selection bias starts to dominate, and the winners.

    Args:
        best: Output of best_by_frame.
        rows: Output of recommendations.

    Returns:
        Left: the smallest subgroup MAE within the budget for each frame, with
        approximately two Monte Carlo standard errors, and the residual
        specimens' absolute selection bias (MAE cannot fall below it).
        Right: the winning design under each priority and selection odds,
        colored by frame.
    """
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(14, 5.2), layout="constrained", width_ratios=(1, 1.15)
    )
    positions = np.arange(len(SELECTION_ODDS))
    for frame, color in FRAME_COLORS.items():
        points = [row for row in best if row["frame"] == frame]
        values = [row["subgroup_mae_pp"] for row in points]
        left.errorbar(
            [row["selection_odds"] for row in points],
            values,
            yerr=[2 * row["mcse_pp"] for row in points],
            color=color,
            marker="o",
            capsize=3,
            label=f"best {frame} design",
        )
    left.plot(
        SELECTION_ODDS,
        [abs(selection_bias_pp(odds)[1]) for odds in SELECTION_ODDS],
        color="#666666",
        linestyle="--",
        label="residual selection bias (absolute)",
    )
    left.set_xticks(SELECTION_ODDS, [f"{odds:g}" for odds in SELECTION_ODDS])
    left.set(
        xlabel="Selection odds for antibody-positive people (residual specimens)",
        ylabel="Subgroup MAE within the budget (points)",
        title="Smallest subgroup error within the budget",
        ylim=(0, None),
    )
    left.legend(loc="upper left", frameon=False)
    left.spines[["top", "right"]].set_visible(False)

    lookup = {(row["priority"], row["selection_odds"]): row for row in rows}
    right.set_xlim(-0.5, len(SELECTION_ODDS) - 0.5)
    right.set_ylim(len(PRIORITIES) - 0.5, -0.5)
    for (r, name), (c, odds) in product(
        enumerate(PRIORITIES), enumerate(SELECTION_ODDS)
    ):
        row = lookup[name, odds]
        right.add_patch(
            plt.Rectangle(
                (c - 0.5, r - 0.5),
                1,
                1,
                facecolor=FRAME_COLORS.get(row["frame"], "#cccccc"),
                edgecolor="white",
                linewidth=1,
            )
        )
        right.text(c, r, row["winner"], ha="center", va="center", color="white")
    right.set_xticks(positions, [f"{odds:g}" for odds in SELECTION_ODDS])
    right.set_yticks(range(len(PRIORITIES)), list(PRIORITIES))
    right.set(xlabel="Selection odds", title="Winning design by priority")
    fig.suptitle(
        "Residual specimens or a community survey?\n"
        "Labels: participants / sites / allocation (P, O) / frame "
        "(C community, R residual); hypothetical costs and bias",
        fontsize=13,
    )
    return fig


def main(argv: Sequence[str] | None = None) -> None:
    """Run the frame grid, save the bundle and the documentation figure.

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
    summary = run_frames(args.reps)
    save_frames(summary, args.out, n_reps=args.reps, overwrite=args.overwrite)
    rows = recommendations(summary)
    best = best_by_frame(summary)
    figure_path = args.figure or (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "assets"
        / "serosurvey_frames.png"
    )
    figure = plot_frames(best, rows)
    figure.savefig(figure_path, dpi=160)
    plt.close(figure)
    print(
        f"{len(summary.configs)} context-design pairs, {args.reps} surveys each, "
        f"{perf_counter() - started:.1f}s; bundle: {args.out}"
    )
    for odds in SELECTION_ODDS:
        other, underserved = selection_bias_pp(odds)
        winners = "; ".join(
            f"{row['priority']}: {row['winner']}"
            for row in rows
            if row["selection_odds"] == odds
        )
        print(f"odds {odds:g} (bias {other:+.1f}/{underserved:+.1f} pp): {winners}")
    for row in best:
        print(
            f"odds {row['selection_odds']:g} {row['frame']}: {row['design']} "
            f"subgroup MAE {row['subgroup_mae_pp']:.2f} ± {2 * row['mcse_pp']:.2f}"
        )


if __name__ == "__main__":
    main()
