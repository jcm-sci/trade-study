"""Deterministic contracts for the hypothetical coastal example."""

from __future__ import annotations

import numpy as np
import pytest

from examples.coastal_observation_design import (
    EVENT_BANK,
    FALSE_ALARM_LIMIT,
    MASTER_TIMES,
    MODE_PARAMETERS,
    NULL_BANK,
    ROOT_SEED,
    SAMPLING_INDICES,
    SyntheticCoastalScorer,
    SyntheticCoastalWorld,
    run_coastal_study,
)
from trade_study import proportion_interval


def test_common_random_numbers_share_truth_and_sampling_schedule() -> None:
    sparse = {
        "measurement_mode": "rapid_sensor",
        "sampling_events": 4,
        "technical_replicates": 1,
    }
    dense = {
        "measurement_mode": "rapid_sensor",
        "sampling_events": 13,
        "technical_replicates": 1,
    }
    world = SyntheticCoastalWorld(EVENT_BANK)
    sparse_truth, sparse_observations = world.generate(sparse, rep=7)
    dense_truth, dense_observations = world.generate(dense, rep=7)
    repeated_truth, repeated_observations = world.generate(sparse, rep=7)

    assert sparse_truth == dense_truth == repeated_truth
    np.testing.assert_array_equal(
        sparse_observations["values"], repeated_observations["values"]
    )
    np.testing.assert_array_equal(
        sparse_observations["values"],
        dense_observations["values"][SAMPLING_INDICES[4]],
    )


def test_technical_replicates_use_nested_error_prefix() -> None:
    config = {
        "measurement_mode": "rapid_sensor",
        "sampling_events": 13,
        "technical_replicates": 3,
    }
    world = SyntheticCoastalWorld(EVENT_BANK)
    truth, observations = world.generate(config, rep=7)
    calibration, errors = world._measurement_bank(  # ruff: ignore[private-member-access]
        rep=7
    )
    latent = truth.amplitude * np.exp(
        -0.5 * ((MASTER_TIMES - truth.center) / truth.width) ** 2
    )
    mode = MODE_PARAMETERS["rapid_sensor"]
    expected = (
        latent
        + mode["bias_scale"] * calibration
        + mode["noise_scale"] * errors[:3].mean(axis=0)
    )
    np.testing.assert_allclose(observations["values"], expected)


def test_cost_is_equal_across_scenario_banks() -> None:
    config = {
        "measurement_mode": "rapid_sensor",
        "sampling_events": 4,
        "technical_replicates": 1,
    }
    event_truth, event_observations = SyntheticCoastalWorld(EVENT_BANK).generate(
        config, rep=7
    )
    null_truth, null_observations = SyntheticCoastalWorld(NULL_BANK).generate(
        config, rep=7
    )
    assert null_truth.amplitude == pytest.approx(0.0)

    scorer = SyntheticCoastalScorer()
    event_score = scorer.score(event_truth, event_observations, config)
    null_score = scorer.score(null_truth, null_observations, config)
    assert event_score["cost_units"] == null_score["cost_units"]


def test_check_study_uses_wilson_gate_and_fixed_three_objective_front() -> None:
    results = run_coastal_study(n_reps=60)
    assert len(results) == 27
    assert {result.event_trials for result in results} == {60}
    assert {result.null_trials for result in results} == {60}
    assert {result.interval_method for result in results} == {"wilson"}

    lookup = {
        (
            result.measurement_mode,
            result.sampling_events,
            result.technical_replicates,
        ): result
        for result in results
    }
    selected = lookup["rapid_sensor", 4, 1]
    assert selected.detection_probability == pytest.approx(40 / 60)
    assert selected.false_alarm_probability == pytest.approx(15 / 60)
    assert selected.event_detections == 40
    assert selected.false_alarms == 15
    assert selected.cost_units == pytest.approx(28.0)
    assert selected.interval_confidence == pytest.approx(0.95)
    assert selected.interval_scope == "fixed-sample-marginal"
    assert selected.root_seed == ROOT_SEED
    interval = proportion_interval(selected.false_alarms, selected.null_trials)
    assert selected.false_alarm_low == pytest.approx(interval.low)
    assert selected.false_alarm_high == pytest.approx(interval.high)
    assert selected.passes_false_alarm_rule == (interval.high <= FALSE_ALARM_LIMIT)

    front = {
        (
            result.measurement_mode,
            result.sampling_events,
            result.technical_replicates,
        )
        for result in results
        if result.pareto_among_evaluated_feasible
    }
    assert front == {
        ("reference_assay", 4, 1),
        ("paired_cross_platform", 4, 1),
        ("reference_assay", 7, 1),
        ("paired_cross_platform", 7, 1),
        ("reference_assay", 13, 1),
        ("paired_cross_platform", 13, 1),
        ("reference_assay", 13, 2),
        ("paired_cross_platform", 13, 2),
        ("paired_cross_platform", 13, 3),
    }
    assert all(
        result.passes_false_alarm_rule
        for result in results
        if (
            result.measurement_mode,
            result.sampling_events,
            result.technical_replicates,
        )
        in front
    )
    assert any(not result.passes_false_alarm_rule for result in results)

    feasible = [result for result in results if result.passes_false_alarm_rule]
    expected_front = set()
    for candidate in feasible:
        dominated = any(
            (
                other.detection_probability >= candidate.detection_probability
                and other.intensity_rmse <= candidate.intensity_rmse
                and other.cost_units <= candidate.cost_units
                and (
                    other.detection_probability > candidate.detection_probability
                    or other.intensity_rmse < candidate.intensity_rmse
                    or other.cost_units < candidate.cost_units
                )
            )
            for other in feasible
        )
        if not dominated:
            expected_front.add((
                candidate.measurement_mode,
                candidate.sampling_events,
                candidate.technical_replicates,
            ))
    assert front == expected_front
