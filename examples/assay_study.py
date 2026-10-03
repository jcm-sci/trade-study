"""Synthetic assay accuracy/cost study with external cost annotations.

All costs and response distributions are invented for this API example;
assay labels do not imply measured clinical performance.

Run: uv run --extra examples python examples/assay_study.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from trade_study import (
    Annotation,
    Direction,
    Factor,
    FactorType,
    Observable,
    Phase,
    Study,
    top_k_pareto_filter,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

UNIT_COST = {"PCR": 18.0, "ELISA": 8.0, "rapid": 3.0}
_NOISE = {"PCR": 0.18, "ELISA": 0.28, "rapid": 0.38}


@dataclass(frozen=True)
class AssayWorld:
    """Invented noisy readouts for binary synthetic truth labels."""

    n_cases: int = 300
    seed: int = 17

    def generate(
        self, config: dict[str, Any], *, rep: int = 0
    ) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
        """Generate a replicate with the same case/noise stream across assays.

        Args:
            config: Assay label and classification threshold.
            rep: Replicate id, offsetting the simulation seed.

        Returns:
            Boolean truth and predicted labels for synthetic cases.
        """
        rng = np.random.default_rng(self.seed + rep)
        truth = rng.random(self.n_cases) < 0.3
        noise = rng.standard_normal(self.n_cases)
        signal = truth.astype(float) + _NOISE[config["assay"]] * noise
        observed = signal >= config["threshold"]
        return truth, observed


class AssayScorer:
    """Return accuracy and unit cost as separate optimization objectives."""

    @staticmethod
    def score(
        truth: NDArray[np.bool_],
        observations: NDArray[np.bool_],
        config: dict[str, Any],
    ) -> dict[str, float]:
        """Score the predictions and promote external cost into an objective.

        Args:
            truth: Synthetic reference labels.
            observations: Synthetic predicted labels.
            config: Assay and threshold values for this evaluation.

        Returns:
            Accuracy to maximize and per-assay cost to minimize.
        """
        return {
            "accuracy": float(np.mean(truth == observations)),
            "cost": UNIT_COST[config["assay"]],
        }


def main() -> None:
    """Screen assays and thresholds, then refine retained designs with more cases.

    Raises:
        RuntimeError: If the study unexpectedly omits cost annotations.
    """
    factors = [
        Factor("assay", FactorType.CATEGORICAL, levels=list(UNIT_COST)),
        Factor("threshold", FactorType.CONTINUOUS, bounds=(0.2, 0.8)),
    ]
    grid = [
        {"assay": assay, "threshold": threshold}
        for assay in UNIT_COST
        for threshold in (0.35, 0.5, 0.65)
    ]
    study = Study(
        world=AssayWorld(),
        scorer=AssayScorer(),
        observables=[
            Observable("accuracy", Direction.MAXIMIZE),
            Observable("cost", Direction.MINIMIZE),
        ],
        factors=factors,
        annotations=[Annotation(name="unit_cost", lookup=UNIT_COST, key="assay")],
        phases=[
            Phase("screen", grid=grid, filter_fn=top_k_pareto_filter(4), n_reps=3),
            Phase("refine", grid="carry", world=AssayWorld(n_cases=3000), n_reps=3),
        ],
    )
    study.run()
    results = study.results("refine").aggregate_replicates()
    print("Refined designs: accuracy, cost objective, external unit-cost annotation")
    if results.annotations is None:
        msg = "Expected unit-cost annotations"
        raise RuntimeError(msg)
    for config, scores, external_values in zip(
        results.configs, results.scores, results.annotations, strict=True
    ):
        print(
            f"{config['assay']:5s} threshold={config['threshold']:.2f} "
            f"accuracy={scores[0]:.3f} cost={scores[1]:.2f} "
            f"unit_cost={external_values[0]:.2f}"
        )


if __name__ == "__main__":
    main()
