"""Compare hypothetical coastal observation designs on fixed scenario banks.

Every signal, error, threshold, and cost in this example is an invented
teaching input. Nothing here is a field measurement, instrument specification,
facility quote, partner commitment, or monitoring recommendation.

Run a fast, deterministic single-process CPU check from the repository root::

    uv run --extra pareto python examples/coastal_observation_design.py --check
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np

from trade_study import (
    Direction,
    Factor,
    FactorType,
    Observable,
    build_grid,
    extract_front,
    run_grid,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

ROOT_SEED = 20261023
DETECTION_THRESHOLD = 0.80
FALSE_ALARM_LIMIT = 0.10
INTERVAL_CONFIDENCE = 0.95
DEFAULT_REPLICATES = 400
CHECK_REPLICATES = 60

# The 13 possible sampling times span a hypothetical normalized monitoring
# window. Each smaller schedule is a fixed subset of the larger schedule.
MASTER_TIMES = np.linspace(0.0, 1.0, 13)
SAMPLING_INDICES: dict[int, NDArray[np.int64]] = {
    4: np.array([0, 4, 8, 12]),
    7: np.array([0, 2, 4, 6, 8, 10, 12]),
    13: np.arange(13),
}

# All values are deliberately hypothetical. ``noise_scale`` and ``bias_scale``
# use the same arbitrary intensity units as the generated event. Cost is an
# invented unit and has no currency conversion.
MODE_PARAMETERS: dict[str, dict[str, float]] = {
    "rapid_sensor": {
        "noise_scale": 0.52,
        "bias_scale": 0.16,
        "fixed_cost": 20.0,
        "sample_cost": 2.0,
    },
    "reference_assay": {
        "noise_scale": 0.22,
        "bias_scale": 0.05,
        "fixed_cost": 35.0,
        "sample_cost": 8.0,
    },
    "paired_cross_platform": {
        "noise_scale": 0.16,
        "bias_scale": 0.06,
        "fixed_cost": 45.0,
        "sample_cost": 10.0,
    },
}


# --8<-- [start:scenario-banks]
@dataclass(frozen=True)
class ScenarioBank:
    """Declare one prespecified synthetic scenario bank."""

    name: str
    event_present: bool


NULL_BANK = ScenarioBank("null", event_present=False)
EVENT_BANK = ScenarioBank("event", event_present=True)
SCENARIO_BANKS = (NULL_BANK, EVENT_BANK)
# --8<-- [end:scenario-banks]


@dataclass(frozen=True)
class SyntheticTruth:
    """Store the latent event parameters for one Monte Carlo replicate."""

    amplitude: float
    center: float
    width: float


@dataclass(frozen=True)
class BankSummary:
    """Store one design's aggregate within one scenario bank."""

    successes: int
    probability: float
    interval_low: float
    interval_high: float
    trials: int
    intensity_rmse: float
    cost_units: float
    interval_confidence: float
    interval_method: str
    interval_scope: str


@dataclass(frozen=True)
class DesignResult:
    """Store the event, null, feasibility, and Pareto summaries for a design."""

    measurement_mode: str
    sampling_events: int
    technical_replicates: int
    detection_probability: float
    detection_low: float
    detection_high: float
    false_alarm_probability: float
    false_alarm_low: float
    false_alarm_high: float
    event_detections: int
    false_alarms: int
    event_trials: int
    null_trials: int
    intensity_rmse: float
    cost_units: float
    interval_confidence: float
    interval_method: str
    interval_scope: str
    root_seed: int
    passes_false_alarm_rule: bool
    pareto_among_evaluated_feasible: bool = False


def stable_seed(*parts: object) -> int:
    """Return a process-independent unsigned seed for a named random stream."""
    payload = json.dumps(
        {"parts": parts, "root_seed": ROOT_SEED},
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


# --8<-- [start:world]
class SyntheticCoastalWorld:
    """Generate shared latent events and nested hypothetical measurement errors.

    Within a scenario bank, a replicate identifier selects the same latent event,
    calibration draw, and standardized error array for every design. Designs
    transform or subset that common random-number bank according to their
    declared settings.
    """

    def __init__(self, scenario: ScenarioBank) -> None:
        """Store the fixed scenario declaration."""
        self.scenario = scenario

    def _truth(self, rep: int) -> SyntheticTruth:
        """Return the shared latent truth for one bank replicate."""
        truth_rng = np.random.default_rng(
            stable_seed(self.scenario.name, rep, "latent-event")
        )
        if self.scenario.event_present:
            amplitude = max(0.20, float(truth_rng.normal(1.20, 0.20)))
            center = float(truth_rng.uniform(0.35, 0.65))
            width = float(truth_rng.uniform(0.07, 0.13))
        else:
            amplitude, center, width = 0.0, 0.50, 0.10
        return SyntheticTruth(amplitude, center, width)

    def _measurement_bank(self, rep: int) -> tuple[float, NDArray[np.float64]]:
        """Return the shared calibration draw and standardized error bank."""
        common_rng = np.random.default_rng(
            stable_seed(self.scenario.name, rep, "measurement-bank")
        )
        return float(common_rng.normal()), common_rng.normal(
            size=(3, MASTER_TIMES.size)
        )

    def generate(
        self, config: dict[str, Any], *, rep: int = 0
    ) -> tuple[SyntheticTruth, dict[str, NDArray[np.float64]]]:
        """Return latent truth and observations for one synthetic replicate.

        Returns:
            Shared latent truth and the design-specific observed values.

        Raises:
            ValueError: If the replicate identifier is negative.
        """
        if rep < 0:
            message = "rep must be non-negative"
            raise ValueError(message)

        truth = self._truth(rep)
        latent = truth.amplitude * np.exp(
            -0.5 * ((MASTER_TIMES - truth.center) / truth.width) ** 2
        )
        calibration_draw, standardized_errors = self._measurement_bank(rep)

        mode = str(config["measurement_mode"])
        parameters = MODE_PARAMETERS[mode]
        technical_replicates = int(config["technical_replicates"])
        sampling_events = int(config["sampling_events"])
        indices = SAMPLING_INDICES[sampling_events]
        measurement_error = standardized_errors[:technical_replicates].mean(axis=0)
        observed = (
            latent
            + parameters["bias_scale"] * calibration_draw
            + parameters["noise_scale"] * measurement_error
        )
        return truth, {"values": observed[indices]}


class SyntheticCoastalScorer:
    """Convert hypothetical observations into detection, error, and cost."""

    def score(
        self,
        truth: SyntheticTruth,
        observations: dict[str, NDArray[np.float64]],
        config: dict[str, Any],
    ) -> dict[str, float]:
        """Score one replicate in declared intensity and invented cost units.

        Returns:
            Detection indicator, squared intensity error, and invented cost.
        """
        estimated_intensity = max(0.0, float(np.max(observations["values"])))
        mode = str(config["measurement_mode"])
        parameters = MODE_PARAMETERS[mode]
        cost = parameters["fixed_cost"] + parameters["sample_cost"] * int(
            config["sampling_events"]
        ) * int(config["technical_replicates"])
        return {
            "detected": float(estimated_intensity >= DETECTION_THRESHOLD),
            "squared_intensity_error": (estimated_intensity - truth.amplitude) ** 2,
            "cost_units": float(cost),
        }


# --8<-- [end:world]


# --8<-- [start:design]
def coastal_design_grid() -> list[dict[str, Any]]:
    """Return the 27-design full factorial teaching grid."""
    factors = [
        Factor(
            "measurement_mode",
            FactorType.CATEGORICAL,
            levels=[
                "rapid_sensor",
                "reference_assay",
                "paired_cross_platform",
            ],
        ),
        Factor("sampling_events", FactorType.DISCRETE, levels=[4, 7, 13]),
        Factor("technical_replicates", FactorType.DISCRETE, levels=[1, 2, 3]),
    ]
    return build_grid(factors, method="full")


def coastal_observables() -> list[Observable]:
    """Return the raw outcomes and their optimization directions."""
    return [
        Observable("detected", Direction.MAXIMIZE),
        Observable("squared_intensity_error", Direction.MINIMIZE),
        Observable("cost_units", Direction.MINIMIZE),
    ]


# --8<-- [end:design]


def _design_key(config: dict[str, Any]) -> tuple[str, int, int]:
    """Return a stable key for joining scenario-bank summaries."""
    return (
        str(config["measurement_mode"]),
        int(config["sampling_events"]),
        int(config["technical_replicates"]),
    )


def _summarize_bank(
    scenario: ScenarioBank, *, n_reps: int, n_jobs: int
) -> dict[tuple[str, int, int], BankSummary]:
    """Return aggregate results for one fixed bank with Wilson intervals."""
    raw = run_grid(
        SyntheticCoastalWorld(scenario),
        SyntheticCoastalScorer(),
        coastal_design_grid(),
        coastal_observables(),
        n_reps=n_reps,
        n_jobs=n_jobs,
    )
    aggregate = raw.aggregate_replicates(
        proportion_observables=["detected"],
        confidence=INTERVAL_CONFIDENCE,
        proportion_method="wilson",
    )
    detection_column = aggregate.observable_names.index("detected")
    error_column = aggregate.observable_names.index("squared_intensity_error")
    cost_column = aggregate.observable_names.index("cost_units")

    summaries: dict[tuple[str, int, int], BankSummary] = {}
    for config, scores, metadata in zip(
        aggregate.configs, aggregate.scores, aggregate.metadata, strict=True
    ):
        interval = metadata["proportion_intervals"]["detected"]
        summaries[_design_key(config)] = BankSummary(
            successes=int(interval["k"]),
            probability=float(scores[detection_column]),
            interval_low=float(interval["low"]),
            interval_high=float(interval["high"]),
            trials=int(interval["n"]),
            intensity_rmse=float(np.sqrt(scores[error_column])),
            cost_units=float(scores[cost_column]),
            interval_confidence=float(interval["confidence"]),
            interval_method=str(interval["method"]),
            interval_scope=str(interval["scope"]),
        )
    return summaries


# --8<-- [start:run]
def run_coastal_study(
    *, n_reps: int = DEFAULT_REPLICATES, n_jobs: int = 1
) -> list[DesignResult]:
    """Evaluate fixed null/event banks and extract a feasible Pareto front.

    The false-alarm gate requires the upper bound of a marginal 95% Wilson
    interval to be no greater than 0.10. The three Pareto objectives among
    designs that pass that rule are event-detection probability (maximize),
    event-intensity root mean squared error (minimize), and hypothetical cost
    units (minimize).

    Args:
        n_reps: Fixed number of Monte Carlo trials per design and scenario.
        n_jobs: Number of CPU workers passed to ``run_grid``.

    Returns:
        All evaluated designs, sorted with feasible Pareto members first.

    Raises:
        ValueError: If fewer than two trials are requested or ``n_jobs`` is zero.
    """
    if n_reps < 2:
        message = "n_reps must be at least 2"
        raise ValueError(message)
    if n_jobs == 0:
        message = "n_jobs must not be zero"
        raise ValueError(message)

    bank_summaries = {
        scenario: _summarize_bank(scenario, n_reps=n_reps, n_jobs=n_jobs)
        for scenario in SCENARIO_BANKS
    }
    event = bank_summaries[EVENT_BANK]
    null = bank_summaries[NULL_BANK]
    results: list[DesignResult] = []
    for config in coastal_design_grid():
        key = _design_key(config)
        event_summary, null_summary = event[key], null[key]
        results.append(
            DesignResult(
                measurement_mode=key[0],
                sampling_events=key[1],
                technical_replicates=key[2],
                detection_probability=event_summary.probability,
                detection_low=event_summary.interval_low,
                detection_high=event_summary.interval_high,
                false_alarm_probability=null_summary.probability,
                false_alarm_low=null_summary.interval_low,
                false_alarm_high=null_summary.interval_high,
                event_detections=event_summary.successes,
                false_alarms=null_summary.successes,
                event_trials=event_summary.trials,
                null_trials=null_summary.trials,
                intensity_rmse=event_summary.intensity_rmse,
                cost_units=event_summary.cost_units,
                interval_confidence=event_summary.interval_confidence,
                interval_method=event_summary.interval_method,
                interval_scope=event_summary.interval_scope,
                root_seed=ROOT_SEED,
                passes_false_alarm_rule=(
                    null_summary.interval_high <= FALSE_ALARM_LIMIT
                ),
            )
        )

    feasible_positions = [
        index for index, result in enumerate(results) if result.passes_false_alarm_rule
    ]
    if feasible_positions:
        objectives = np.array(
            [
                [
                    results[index].detection_probability,
                    results[index].intensity_rmse,
                    results[index].cost_units,
                ]
                for index in feasible_positions
            ],
            dtype=float,
        )
        local_front = extract_front(
            objectives,
            [Direction.MAXIMIZE, Direction.MINIMIZE, Direction.MINIMIZE],
        )
        for local_index in local_front:
            position = feasible_positions[int(local_index)]
            results[position] = replace(
                results[position], pareto_among_evaluated_feasible=True
            )

    return sorted(
        results,
        key=lambda result: (
            not result.pareto_among_evaluated_feasible,
            result.cost_units,
            -result.detection_probability,
            result.measurement_mode,
            result.sampling_events,
            result.technical_replicates,
        ),
    )


# --8<-- [end:run]


def _print_results(results: list[DesignResult], *, n_reps: int) -> None:
    """Print the feasible-front projection and finite-bank counts."""
    feasible = [result for result in results if result.passes_false_alarm_rule]
    front = [result for result in results if result.pareto_among_evaluated_feasible]
    print(
        f"Hypothetical study: {len(results)} designs, {n_reps} fixed trials "
        f"per design and scenario, root seed {ROOT_SEED}, "
        f"{len(feasible)} feasible, {len(front)} on front"
    )
    print(
        "mode                    events tech  detection k/n (95% Wilson)  "
        "false alarm k/n (95% Wilson)  RMSE   cost"
    )
    for result in front:
        print(
            f"{result.measurement_mode:24} "
            f"{result.sampling_events:>6} {result.technical_replicates:>4}  "
            f"{result.detection_probability:>5.3f} "
            f"{result.event_detections:>3}/{result.event_trials:<3} "
            f"[{result.detection_low:>5.3f}, {result.detection_high:>5.3f}]  "
            f"{result.false_alarm_probability:>5.3f} "
            f"{result.false_alarms:>3}/{result.null_trials:<3} "
            f"[{result.false_alarm_low:>5.3f}, {result.false_alarm_high:>5.3f}]  "
            f"{result.intensity_rmse:>5.3f}  {result.cost_units:>5.1f}"
        )


def main() -> None:
    """Run the full teaching study or its fast single-process CPU check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "run the complete workflow with 60 trials per design and scenario "
            "on one CPU process"
        ),
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help=(
            "parallel CPU workers for the full run (requires the parallel extra); "
            "--check always uses one"
        ),
    )
    args = parser.parse_args()
    n_reps = CHECK_REPLICATES if args.check else DEFAULT_REPLICATES
    n_jobs = 1 if args.check else args.n_jobs
    results = run_coastal_study(n_reps=n_reps, n_jobs=n_jobs)
    _print_results(results, n_reps=n_reps)


if __name__ == "__main__":
    main()
