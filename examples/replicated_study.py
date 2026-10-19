"""Compare replicated sample-mean designs with common random numbers.

Run with ``python examples/replicated_study.py`` after installing the
``pareto`` extra. The intervals summarize Monte Carlo error, not gate power.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from trade_study import (
    Direction,
    Factor,
    FactorType,
    Observable,
    build_grid,
    extract_front,
    paired_difference,
    run_grid,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from trade_study import ResultsTable


class SampleMean:
    """Estimate a zero mean using nested samples from a common replicate bank."""

    def generate(
        self, config: dict[str, Any], *, rep: int = 0
    ) -> tuple[float, NDArray[np.float64]]:
        """Return truth and the selected prefix of the replicate's sample bank."""
        bank = np.random.default_rng(np.random.SeedSequence([2026, rep])).normal(
            size=32
        )
        return 0.0, bank[: int(config["samples"])]


class ErrorAndCost:
    """Report squared error and a deterministic sample cost."""

    def score(
        self, truth: float, observed: NDArray[np.float64], config: dict[str, Any]
    ) -> dict[str, float]:
        """Return objectives in the declared observable order."""
        return {
            "error": float((observed.mean() - truth) ** 2),
            "cost": float(config["samples"]),
        }


def run_example() -> ResultsTable:
    """Run a fixed bank with paired error and cost objectives.

    Returns:
        The per-replicate table, retaining design and replicate identities.
    """
    grid = build_grid(
        [Factor("samples", FactorType.DISCRETE, levels=[8, 32])], method="full"
    )
    observables = [
        Observable("error", Direction.MINIMIZE),
        Observable("cost", Direction.MINIMIZE),
    ]
    results = run_grid(SampleMean(), ErrorAndCost(), grid, observables, n_reps=64)
    means = results.aggregate_replicates()
    front = extract_front(means.scores, [o.direction for o in observables])
    contrast = paired_difference(
        results, {"samples": 32}, {"samples": 8}, "error", seed=2026
    )
    print({
        "replicate_rows": len(results.configs),
        "front": front.tolist(),
        "paired_error_difference": contrast.mean,
        "finite_pairs": contrast.n_pairs,
    })
    return results


if __name__ == "__main__":
    run_example()
