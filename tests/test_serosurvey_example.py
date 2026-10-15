"""Statistical and decision invariants for the synthetic serosurvey demo."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pytest

from examples import serosurvey_study
from examples.serosurvey_study import (
    EQUIVALENCE,
    GROUP_WEIGHTS,
    INDICATOR,
    PRECISION_TARGET_PP,
    PRIORITIES,
    REFINE_REPS,
    TARGET_HALF_WIDTH,
    SurveyCosts,
    SurveyOutcome,
    SurveyScorer,
    SurveyWorld,
    acceptable_sets,
    allocation_counts,
    planned_replicates,
    plot_precision,
    plot_priorities,
    plot_robustness,
    precision_rows,
    preference_policy,
    robustness_rows,
    run_study,
    survey_factors,
    survey_indicators,
    survey_observables,
)
from trade_study import (
    Constraint,
    PreferenceSweep,
    ResultsTable,
    build_grid,
    preference_sweep,
    proportion_interval,
    replicates_for_half_width,
    run_grid,
)


def test_allocations_preserve_exact_totals() -> None:
    grid = build_grid(survey_factors(), method="full")
    assert len(grid) == 18
    for config in grid:
        people, communities = allocation_counts(config)
        assert sum(people) == config["participants"]
        assert sum(communities) == config["communities"]
        expected_share = 0.2 if config["allocation"] == "proportional" else 0.5
        assert people[1] / sum(people) == pytest.approx(expected_share)
        assert communities[1] / sum(communities) == pytest.approx(expected_share)


def test_score_uses_population_weights_when_oversampling() -> None:
    config = {"participants": 300, "communities": 10, "allocation": "oversample"}
    truth = np.array([0.9, 0.65])
    outcome = SurveyOutcome(np.array([0.9, 0.55]), (150, 150), (5, 5))
    scores = SurveyScorer().score(truth, outcome, config)
    assert scores["overall_error_pp"] == pytest.approx(2.0)
    assert scores["underserved_error_pp"] == pytest.approx(10.0)
    assert scores["cost_usd"] == 18_250
    assert np.dot(GROUP_WEIGHTS, truth) == pytest.approx(0.85)


@pytest.mark.parametrize("correlation", [0.0, 0.3])
def test_deterministic_population_has_zero_error(correlation: float) -> None:
    world = SurveyWorld(prevalence=(1.0, 0.0), correlation=correlation)
    config = {"participants": 300, "communities": 40, "allocation": "proportional"}
    truth, outcome = world.generate(config, rep=7)
    np.testing.assert_array_equal(outcome.prevalence, truth)
    scores = SurveyScorer().score(truth, outcome, config)
    assert scores["overall_error_pp"] == 0
    assert scores["underserved_error_pp"] == 0


def test_clustered_estimator_matches_known_mean_and_variance() -> None:
    world = SurveyWorld(prevalence=(0.9, 0.65), correlation=0.08)
    config = {"participants": 300, "communities": 40, "allocation": "proportional"}
    estimates = np.array([
        world.generate(config, rep=rep)[1].prevalence for rep in range(5000)
    ])
    # Unequal cluster sizes are intentional: 60 people / 8 communities
    # requires four clusters of size 8 and four of size 7.
    n, sum_squares, p, rho = 60, 4 * 8**2 + 4 * 7**2, 0.65, 0.08
    variance = p * (1 - p) * ((1 - rho) / n + rho * sum_squares / n**2)
    assert abs(estimates[:, 1].mean() - p) < 4 * np.sqrt(variance / len(estimates))
    assert estimates[:, 1].var(ddof=1) == pytest.approx(variance, rel=0.12)


def test_streams_replay_and_phase_seeds_are_distinct() -> None:
    config = {"participants": 300, "communities": 10, "allocation": "proportional"}
    first = SurveyWorld(seed=2026).generate(config, rep=3)[1].prevalence
    replay = SurveyWorld(seed=2026).generate(config, rep=3)[1].prevalence
    new_phase = SurveyWorld(seed=2027).generate(config, rep=3)[1].prevalence
    new_replicate = SurveyWorld(seed=2026).generate(config, rep=4)[1].prevalence
    np.testing.assert_array_equal(first, replay)
    assert not np.array_equal(first, new_phase)
    assert not np.array_equal(first, new_replicate)


@pytest.mark.parametrize(
    "settings",
    [
        {"prevalence": (0.9, np.nan)},
        {"prevalence": (0.9, 1.2)},
        {"correlation": -0.1},
        {"correlation": 1.0},
        {"seed": -1},
    ],
)
def test_invalid_model_assumptions_are_rejected(settings: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="Require"):
        SurveyWorld(**settings)


@pytest.mark.parametrize(
    "config",
    [
        {"participants": 300.0, "communities": 10, "allocation": "proportional"},
        {"participants": 300, "communities": 40, "allocation": "unknown"},
        {"participants": 30, "communities": 40, "allocation": "proportional"},
        {"participants": 300, "communities": 11, "allocation": "proportional"},
    ],
)
def test_invalid_designs_are_rejected(config: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match=r"integers|Allocation|Use integer"):
        SurveyWorld().generate(config)


def test_budget_change_uses_existing_results_and_reports_no_choice() -> None:
    grid = build_grid(survey_factors(), method="full")[:2]
    observables = survey_observables()
    raw = run_grid(SurveyWorld(), SurveyScorer(), grid, observables, n_reps=8)
    original_scores = raw.scores.copy()
    for budget, feasible in [(16_000, [True, False]), (1, [False, False])]:
        report = preference_sweep(
            raw,
            observables,
            policy=preference_policy(),
            constraints=[Constraint("budget", "cost_usd", "<=", budget)],
        )
        assert report.feasible.tolist() == feasible
        if budget == 1:
            assert np.all(np.isnan(report.ranks))
            assert not report.pareto.any()
            figure = plot_priorities(report)
            assert figure.axes[0].texts[0].get_text() == "No design meets the budget"
            plt.close(figure)
    np.testing.assert_array_equal(raw.scores, original_scores)


def test_financial_costs_are_additive_and_nonnegative() -> None:
    config = {"participants": 300, "communities": 10, "allocation": "proportional"}
    components = SurveyCosts().components(config)
    assert components == {"Setup": 3000, "Community visits": 3400, "Participants": 9600}
    assert sum(components.values()) == 16_000
    with pytest.raises(ValueError, match="finite and nonnegative"):
        SurveyCosts(setup=-1)


def _sweep(budget: float) -> PreferenceSweep:
    grid = build_grid(survey_factors(), method="full")
    observables = survey_observables()
    raw = run_grid(SurveyWorld(), SurveyScorer(), grid, observables, n_reps=8)
    return preference_sweep(
        raw,
        observables,
        policy=preference_policy(),
        constraints=[Constraint("budget", "cost_usd", "<=", budget)],
    )


def test_policy_declares_positive_raw_unit_tolerances() -> None:
    policy = preference_policy()
    assert policy.equivalence == EQUIVALENCE
    assert set(EQUIVALENCE) == set(PRIORITIES["Cost first"])
    assert all(value > 0 for value in EQUIVALENCE.values())


def test_notebook_shows_the_same_tolerances() -> None:
    notebook = Path(__file__).resolve().parents[1] / "examples/serosurvey_study.ipynb"
    cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    source = next(
        "".join(cell["source"])
        for cell in cells
        if cell["cell_type"] == "code" and "equivalence=" in "".join(cell["source"])
    )
    block = re.search(r"equivalence=(\{.*?\})", source, re.DOTALL)
    assert block is not None
    assert ast.literal_eval(block.group(1)) == EQUIVALENCE


def test_acceptable_sets_hold_winners_within_tolerance() -> None:
    report = _sweep(40_000)
    sets = acceptable_sets(report)
    assert list(sets) == list(PRIORITIES)
    names = report.summary.observable_names
    for ranks, members in zip(report.ranks, sets.values(), strict=True):
        winners = np.flatnonzero(ranks == 1)
        assert set(winners) <= set(members)
        assert report.feasible[members].all()
        for member in members:
            gaps = np.abs(
                report.summary.scores[member] - report.summary.scores[winners]
            )
            tolerance = np.array([EQUIVALENCE[name] for name in names])
            assert np.any(np.all(gaps <= tolerance, axis=1))


def test_robustness_rows_are_sorted_and_consistent_with_ranks() -> None:
    report = _sweep(40_000)
    rows = robustness_rows(report)
    assert len(rows) == int(report.pareto.sum())
    costs = [row["cost_usd"] for row in rows]
    assert costs == sorted(costs)
    for row in rows:
        index = int(row["design"][1:]) - 1
        assert report.pareto[index]
        assert row["max_regret"] >= 0
        for name, ranks in zip(PRIORITIES, report.ranks, strict=True):
            assert (name in row["wins"]) == (ranks[index] == 1)
            if ranks[index] == 1:
                assert name in row["acceptable_for"]
    figure = plot_robustness(report)
    assert len(figure.axes[0].patches) == len(rows)
    plt.close(figure)


def test_infeasible_budget_has_no_robustness_rows() -> None:
    report = _sweep(1)
    assert robustness_rows(report) == []
    assert all(members == [] for members in acceptable_sets(report).values())
    figure = plot_robustness(report)
    assert figure.axes[0].texts[0].get_text() == "No design meets the budget"
    plt.close(figure)


@pytest.mark.parametrize(
    ("estimate", "expected"), [(0.60, 1.0), (0.70, 1.0), (0.5999, 0.0), (0.71, 0.0)]
)
def test_precision_indicator_is_inclusive_at_the_target(
    estimate: float, expected: float
) -> None:
    config = {"participants": 300, "communities": 10, "allocation": "oversample"}
    truth = np.array([0.9, 0.65])
    outcome = SurveyOutcome(np.array([0.9, estimate]), (150, 150), (5, 5))
    scores = SurveyScorer().score(truth, outcome, config)
    assert PRECISION_TARGET_PP == 5
    assert scores[INDICATOR] == expected


def test_planned_replicates_fit_the_refinement_phase() -> None:
    planned = planned_replicates()
    assert planned == replicates_for_half_width(TARGET_HALF_WIDTH)
    assert planned <= REFINE_REPS
    worst = proportion_interval(REFINE_REPS // 2, REFINE_REPS)
    assert (worst.high - worst.low) / 2 <= TARGET_HALF_WIDTH


def test_run_study_refuses_too_few_replicates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(serosurvey_study, "REFINE_REPS", 10)
    with pytest.raises(ValueError, match="planned indicator precision"):
        run_study(SurveyWorld())


def _raw_with_indicator(n_reps: int) -> ResultsTable:
    grid = build_grid(survey_factors(), method="full")
    observables = survey_observables() + survey_indicators()
    return run_grid(SurveyWorld(), SurveyScorer(), grid, observables, n_reps=n_reps)


def test_precision_rows_match_raw_indicator_counts() -> None:
    raw = _raw_with_indicator(40)
    summary = raw.aggregate_replicates(proportion_observables=[INDICATOR])
    report = preference_sweep(
        raw,
        survey_observables(),
        policy=preference_policy(),
        constraints=[Constraint("budget", "cost_usd", "<=", 40_000)],
    )
    rows = precision_rows(report, summary)
    assert len(rows) == int(report.pareto.sum())
    costs = [row["cost_usd"] for row in rows]
    assert costs == sorted(costs)
    column = raw.observable_names.index(INDICATOR)
    for row in rows:
        config = {
            key: row[key] for key in ("participants", "communities", "allocation")
        }
        matches = [i for i, c in enumerate(raw.configs) if c == config]
        assert row["n"] == len(matches) == 40
        assert row["k"] == int(raw.scores[matches, column].sum())
        assert row["low"] <= row["within_target"] <= row["high"]
        assert row["within_target"] == pytest.approx(row["k"] / row["n"])
    figure = plot_precision(report, summary)
    assert len(figure.axes[0].patches) == len(rows)
    plt.close(figure)


def test_infeasible_budget_has_no_precision_rows() -> None:
    raw = _raw_with_indicator(8)
    summary = raw.aggregate_replicates(proportion_observables=[INDICATOR])
    report = preference_sweep(
        raw,
        survey_observables(),
        policy=preference_policy(),
        constraints=[Constraint("budget", "cost_usd", "<=", 1)],
    )
    assert precision_rows(report, summary) == []
    figure = plot_precision(report, summary)
    assert figure.axes[0].texts[0].get_text() == "No design meets the budget"
    plt.close(figure)
