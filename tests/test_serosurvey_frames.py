"""Sampling-frame comparison for the synthetic serosurvey demo (#191)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import pytest

from examples.serosurvey_frames import (
    BUNDLE_DIR,
    CONTEXT_KEYS,
    DESIGN_KEYS,
    FRAME_SEED,
    SELECTION_ODDS,
    FrameScorer,
    FrameWorld,
    best_by_frame,
    design_label,
    frame_grid,
    plot_frames,
    recommendations,
    run_frames,
    save_frames,
    selection_bias_pp,
    split_config,
)
from examples.serosurvey_study import (
    BUDGET,
    INDICATOR,
    PRIORITIES,
    REFINE_REPS,
    SurveyCosts,
    SurveyScorer,
    SurveyWorld,
    allocation_counts,
    run_study,
    sampling_frame,
    survey_factors,
    survey_indicators,
    survey_observables,
)
from trade_study import build_grid, load_result_bundle

if TYPE_CHECKING:
    from pathlib import Path

    from trade_study import ResultsBundle

RESIDUAL = {
    "participants": 600,
    "communities": 20,
    "allocation": "oversample",
    "frame": "residual",
}


@pytest.fixture(scope="module")
def bundle() -> ResultsBundle:
    return load_result_bundle(BUNDLE_DIR)


def test_unbiased_residual_specimens_reproduce_the_community_survey() -> None:
    world = SurveyWorld()
    for config in build_grid(survey_factors(), method="full"):
        for rep in range(3):
            truth, community = world.generate(config, rep=rep)
            residual_truth, residual = world.generate(
                {**config, "frame": "residual"}, rep=rep
            )
            np.testing.assert_array_equal(residual_truth, truth)
            np.testing.assert_array_equal(residual.prevalence, community.prevalence)


def test_selection_shifts_each_community_by_the_odds() -> None:
    odds = 1.5
    world = SurveyWorld(selection_odds=(odds, odds))
    people, communities = allocation_counts(RESIDUAL)
    prevalences, draws = world.population(4)
    _, outcome = world.generate(RESIDUAL, rep=4)
    for group, (pi, uniforms, n, k) in enumerate(
        zip(prevalences, draws, people, communities, strict=True)
    ):
        shifted = odds * pi[:k] / (odds * pi[:k] + 1 - pi[:k])
        positives = sum(
            int((uniforms[j, : n // k + (j < n % k)] < shifted[j]).sum())
            for j in range(k)
        )
        assert outcome.prevalence[group] == pytest.approx(positives / n)
    # Same draws, higher prevalence among specimens: never fewer positives.
    _, community = world.generate({**RESIDUAL, "frame": "community"}, rep=4)
    assert np.all(outcome.prevalence >= community.prevalence)


def test_residual_bias_matches_the_selection_model() -> None:
    odds = 1.25
    world = SurveyWorld(seed=FRAME_SEED, selection_odds=(odds, odds))
    errors = []
    for rep in range(1000):
        truth, outcome = world.generate(RESIDUAL, rep=rep)
        errors.append(100 * (outcome.prevalence - truth))
    errors_pp = np.array(errors)
    expected = np.array(selection_bias_pp(odds))
    tolerance = 4 * errors_pp.std(axis=0, ddof=1) / np.sqrt(len(errors_pp))
    assert np.all(np.abs(errors_pp.mean(axis=0) - expected) < tolerance + 0.05)


def test_selection_bias_is_zero_without_selection_and_grows() -> None:
    assert selection_bias_pp(1.0, reps=5) == pytest.approx((0.0, 0.0), abs=1e-12)
    underserved = [selection_bias_pp(odds, reps=20)[1] for odds in SELECTION_ODDS]
    assert underserved == sorted(underserved)
    other, subgroup = selection_bias_pp(1.25, reps=20)
    assert 0 < other < subgroup


def test_residual_costs_replace_community_visits() -> None:
    costs = SurveyCosts()
    for config in build_grid(survey_factors(), method="full"):
        community = costs.components(config)
        residual = costs.components({**config, "frame": "residual"})
        people, communities = allocation_counts(config)
        assert list(residual) == ["Setup", "Facilities", "Specimens"]
        assert residual["Facilities"] == pytest.approx(
            np.dot(communities, costs.per_facility)
        )
        assert residual["Specimens"] == pytest.approx(
            np.dot(people, costs.per_specimen)
        )
        assert sum(residual.values()) < sum(community.values())
    smallest = {"participants": 300, "communities": 10, "allocation": "proportional"}
    assert sum(costs.components({**smallest, "frame": "residual"}).values()) == 7_980


@pytest.mark.parametrize(
    "odds", [(1.0,), (1.0, 0.0), (1.0, -1.0), (1.0, np.inf), (1.0, np.nan)]
)
def test_invalid_selection_odds_are_rejected(odds: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="selection odds"):
        SurveyWorld(selection_odds=odds)  # type: ignore[arg-type]


def test_invalid_frames_and_costs_are_rejected() -> None:
    assert sampling_frame({}) == "community"
    clinic = {**RESIDUAL, "frame": "clinic"}
    with pytest.raises(ValueError, match="Frame"):
        sampling_frame(clinic)
    with pytest.raises(ValueError, match="Frame"):
        SurveyWorld().generate(clinic)
    with pytest.raises(ValueError, match="Frame"):
        SurveyCosts().components(clinic)
    with pytest.raises(ValueError, match="one cost per population group"):
        SurveyCosts(per_facility=(100.0,))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="nonnegative"):
        SurveyCosts(per_specimen=(12.0, -1.0))


def test_frame_grid_keeps_context_and_design_separate() -> None:
    grid = frame_grid()
    assert len(grid) == len(SELECTION_ODDS) * 36
    for config in grid:
        context, design = split_config(config)
        assert tuple(context) == CONTEXT_KEYS
        assert tuple(design) == DESIGN_KEYS
    with pytest.raises(KeyError, match="lacks"):
        split_config({"participants": 300})
    assert design_label(RESIDUAL) == "600/20/O/R"


def test_frame_world_applies_the_odds_to_both_groups() -> None:
    config: dict[str, Any] = {"selection_odds": 1.25, **RESIDUAL}
    _, design = split_config(config)
    expected = SurveyWorld(seed=FRAME_SEED, selection_odds=(1.25, 1.25)).generate(
        design, rep=2
    )
    truth, outcome = FrameWorld().generate(config, rep=2)
    np.testing.assert_array_equal(truth, expected[0])
    np.testing.assert_array_equal(outcome.prevalence, expected[1].prevalence)
    assert FrameScorer().score(truth, outcome, config) == SurveyScorer().score(
        truth, outcome, design
    )


def test_run_frames_refuses_too_few_replicates() -> None:
    with pytest.raises(ValueError, match="planned indicator precision"):
        run_frames(10)


def test_committed_bundle_is_valid_and_documented(bundle: ResultsBundle) -> None:
    results = bundle.results
    assert len(results.configs) == len(frame_grid())
    assert [o.name for o in bundle.observables] == [
        o.name for o in survey_observables() + survey_indicators()
    ]
    caller = bundle.provenance["caller"]
    assert caller["seed"] == FRAME_SEED
    assert caller["replicates_per_design"] == REFINE_REPS
    assert caller["contexts"]["selection_odds"] == list(SELECTION_ODDS)
    assert caller["costs"]["per_specimen"] == list(SurveyCosts().per_specimen)
    assert caller["hypothetical"] is True
    for meta in results.metadata:
        interval = meta["proportion_intervals"][INDICATOR]
        assert interval["n"] == REFINE_REPS
        assert interval["low"] <= interval["estimate"] <= interval["high"]


def test_community_designs_reproduce_the_main_study(bundle: ResultsBundle) -> None:
    _, summary, _ = run_study(SurveyWorld(seed=2026))
    expected = {
        tuple(sorted(config.items())): row
        for config, row in zip(summary.configs, summary.scores, strict=True)
    }
    community = [
        (split_config(config)[1], row)
        for config, row in zip(
            bundle.results.configs, bundle.results.scores, strict=True
        )
        if config["frame"] == "community"
    ]
    assert len(community) == len(SELECTION_ODDS) * 18
    for design, row in community:
        key = tuple(sorted((k, v) for k, v in design.items() if k != "frame"))
        np.testing.assert_allclose(row, expected[key], rtol=1e-12)


def test_recommendations_show_where_bias_dominates(bundle: ResultsBundle) -> None:
    rows = recommendations(bundle.results)
    assert len(rows) == len(SELECTION_ODDS) * len(PRIORITIES)
    assert SELECTION_ODDS[0] == pytest.approx(1)
    unbiased = [row for row in rows if row["selection_odds"] == SELECTION_ODDS[0]]
    # Without selection bias, each residual design matches its community twin's
    # accuracy at a lower cost, so residual specimens win every priority.
    assert {row["frame"] for row in unbiased} == {"residual"}
    strongest = {
        row["priority"]: row["frame"]
        for row in rows
        if row["selection_odds"] == max(SELECTION_ODDS)
    }
    assert strongest["Subgroup accuracy"] == "community"
    for row in rows:
        assert row["low"] <= row["within_target"] <= row["high"]

    best = best_by_frame(bundle.results)
    assert len(best) == 2 * len(SELECTION_ODDS)
    community = {row["subgroup_mae_pp"] for row in best if row["frame"] == "community"}
    assert len(community) == 1
    for row in best:
        assert row["cost_usd"] <= BUDGET
        # Mean absolute error cannot fall below the absolute bias.
        assert row["subgroup_mae_pp"] >= abs(row["bias_pp"]) - 3 * row["mcse_pp"]
    figure = plot_frames(best, rows)
    assert len(figure.axes) == 2
    assert len(figure.axes[1].texts) == len(rows)
    plt.close(figure)


def test_save_frames_round_trip_and_overwrite(
    bundle: ResultsBundle, tmp_path: Path
) -> None:
    destination = tmp_path / "bundle"
    results = bundle.results
    save_frames(results, destination, n_reps=REFINE_REPS)
    loaded = load_result_bundle(destination)
    np.testing.assert_array_equal(loaded.results.scores, results.scores)
    assert loaded.results.configs == results.configs
    with pytest.raises(FileExistsError):
        save_frames(results, destination, n_reps=REFINE_REPS)
    save_frames(results, destination, n_reps=REFINE_REPS, overwrite=True)
    assert load_result_bundle(destination).results.configs == results.configs
