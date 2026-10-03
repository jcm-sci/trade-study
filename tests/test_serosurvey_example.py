"""Statistical and decision invariants for the synthetic serosurvey demo."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pytest

from examples.serosurvey_study import (
    GROUP_WEIGHTS,
    SurveyCosts,
    SurveyOutcome,
    SurveyScorer,
    SurveyWorld,
    allocation_counts,
    plot_priorities,
    preference_policy,
    survey_factors,
    survey_observables,
)
from trade_study import Constraint, build_grid, preference_sweep, run_grid


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
