"""Count actual Sobol callback work rather than unique parameter configurations."""

from __future__ import annotations

import pytest

from trade_study import Factor, FactorType, screen, sobol_indices


@pytest.mark.parametrize("entry_point", ["screen", "sobol_indices"])
@pytest.mark.parametrize("dimensions", [1, 3])
@pytest.mark.parametrize("replicates", [1, 2])
def test_sobol_nominal_configurations_and_physical_calls(
    entry_point: str, dimensions: int, replicates: int
) -> None:
    """Both APIs retain N(2D+2) rows and invoke every requested replicate."""
    factors = [
        Factor(f"x{i}", FactorType.CONTINUOUS, bounds=(0, 1)) for i in range(dimensions)
    ]
    calls: list[tuple[tuple[float, ...], int]] = []

    def evaluate(config: dict[str, float], *, rep: int = 0) -> dict[str, float]:
        values = tuple(config[f.name] for f in factors)
        calls.append((values, rep))
        return {"loss": sum((i + 1) * value**2 for i, value in enumerate(values))}

    if entry_point == "screen":
        screen(
            evaluate,
            factors,
            method="sobol",
            n_trajectories=8,
            n_reps=replicates,
            seed=5,
        )
    else:
        sobol_indices(evaluate, factors, n_samples=8, n_reps=replicates, seed=5)

    nominal_rows = 8 * (2 * dimensions + 2)
    assert len(calls) == nominal_rows * replicates
    assert len(calls[::replicates]) == nominal_rows
    if dimensions == 1:
        assert len({config for config, _ in calls}) < nominal_rows
    for start in range(0, len(calls), replicates):
        batch = calls[start : start + replicates]
        assert [rep for _, rep in batch] == list(range(replicates))
        assert all(config == batch[0][0] for config, _ in batch)
