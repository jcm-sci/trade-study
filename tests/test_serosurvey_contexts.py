"""Context grid for the synthetic serosurvey demo (#188)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pytest

from examples import serosurvey_contexts as contexts
from examples.serosurvey_contexts import (
    BASELINE,
    BUNDLE_DIR,
    CONTEXT_KEYS,
    CONTEXT_SEED,
    DESIGN_KEYS,
    ContextScorer,
    ContextWorld,
    context_grid,
    context_results,
    plot_contexts,
    recommendations,
    save_contexts,
    sensitivity,
    split_config,
)
from examples.serosurvey_study import (
    INDICATOR,
    PRIORITIES,
    REFINE_REPS,
    SurveyScorer,
    SurveyWorld,
    run_study,
    survey_indicators,
    survey_observables,
)
from trade_study import load_result_bundle

if TYPE_CHECKING:
    from pathlib import Path

    from trade_study import ResultsBundle


@pytest.fixture(scope="module")
def bundle() -> ResultsBundle:
    return load_result_bundle(BUNDLE_DIR)


def test_context_grid_keeps_context_and_design_separate() -> None:
    grid = context_grid()
    assert len(grid) == 9 * 18
    assert len({tuple(sorted(config.items())) for config in grid}) == len(grid)
    for config in grid:
        context, design = split_config(config)
        assert set(context) == set(CONTEXT_KEYS)
        assert set(design) == set(DESIGN_KEYS)
    with pytest.raises(KeyError, match="lacks"):
        split_config({"participants": 300})


def test_context_world_delegates_to_survey_world() -> None:
    config = {
        "correlation": 0.12,
        "underserved_prevalence": 0.55,
        "participants": 600,
        "communities": 20,
        "allocation": "oversample",
    }
    _, design = split_config(config)
    expected = SurveyWorld(
        prevalence=(0.9, 0.55), correlation=0.12, seed=CONTEXT_SEED
    ).generate(design, rep=3)
    truth, outcome = ContextWorld().generate(config, rep=3)
    np.testing.assert_array_equal(truth, expected[0])
    np.testing.assert_array_equal(outcome.prevalence, expected[1].prevalence)
    assert ContextScorer().score(truth, outcome, config) == SurveyScorer().score(
        truth, outcome, design
    )


def test_run_contexts_refuses_too_few_replicates() -> None:
    with pytest.raises(ValueError, match="planned indicator precision"):
        contexts.run_contexts(10)


def test_committed_bundle_is_valid_and_documented(bundle: ResultsBundle) -> None:
    results = bundle.results
    assert len(results.configs) == 162
    assert [o.name for o in bundle.observables] == [
        o.name for o in survey_observables() + survey_indicators()
    ]
    caller = bundle.provenance["caller"]
    assert caller["seed"] == CONTEXT_SEED
    assert caller["replicates_per_design"] == REFINE_REPS
    assert caller["hypothetical"] is True
    for meta in results.metadata:
        interval = meta["proportion_intervals"][INDICATOR]
        assert interval["n"] == REFINE_REPS
        assert interval["low"] <= interval["estimate"] <= interval["high"]


def test_baseline_context_reproduces_the_main_study(bundle: ResultsBundle) -> None:
    baseline = context_results(bundle.results, BASELINE)
    _, summary, _ = run_study(SurveyWorld(seed=2026))
    expected = {
        tuple(sorted(config.items())): row
        for config, row in zip(summary.configs, summary.scores, strict=True)
    }
    assert len(baseline.configs) == len(expected) == 18
    for config, row in zip(baseline.configs, baseline.scores, strict=True):
        np.testing.assert_allclose(
            row, expected[tuple(sorted(config.items()))], rtol=1e-12
        )


def test_recommendations_cover_every_context_and_priority(
    bundle: ResultsBundle,
) -> None:
    rows = recommendations(bundle.results)
    assert len(rows) == 9 * len(PRIORITIES)
    baseline = {
        row["priority"]: row["winner"]
        for row in rows
        if all(row[key] == value for key, value in BASELINE.items())
    }
    assert baseline == {
        "Cost first": "300/10/P",
        "Overall accuracy": "600/40/P",
        "Subgroup accuracy": "600/20/O",
    }
    for row in rows:
        assert row["low"] <= row["within_target"] <= row["high"]
    counts = sensitivity(rows)
    assert set(counts) == set(PRIORITIES)
    for value in counts.values():
        assert 0 <= value["correlation"] <= 6
        assert 0 <= value["underserved_prevalence"] <= 6
    figure = plot_contexts(rows)
    assert len(figure.axes) == len(PRIORITIES)
    plt.close(figure)


def test_save_contexts_round_trip_and_overwrite(
    bundle: ResultsBundle, tmp_path: Path
) -> None:
    destination = tmp_path / "bundle"
    results = bundle.results
    save_contexts(results, destination, n_reps=REFINE_REPS)
    loaded = load_result_bundle(destination)
    np.testing.assert_array_equal(loaded.results.scores, results.scores)
    assert loaded.results.configs == results.configs
    with pytest.raises(FileExistsError):
        save_contexts(results, destination, n_reps=REFINE_REPS)
    save_contexts(results, destination, n_reps=REFINE_REPS, overwrite=True)
    assert load_result_bundle(destination).results.configs == results.configs
