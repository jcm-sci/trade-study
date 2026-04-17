"""CSTR reactor design study.

A multi-phase trade study over a continuous stirred-tank reactor (CSTR)
with competing objectives: conversion, selectivity, and energy cost.

The reactor model uses Arrhenius kinetics as closed-form ground truth —
no external dependencies beyond numpy.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from trade_study import (
    Direction,
    Factor,
    FactorType,
    Observable,
    Phase,
    Study,
    build_grid,
    extract_front,
    plot_front,
    plot_parallel,
    plot_scores,
    top_k_pareto_filter,
)

ASSET_DIR = "docs/assets"

# ── Reactor kinetics (ground truth) ────────────────────────────────

# --8<-- [start:kinetics]
# A -> B -> C  (series reactions, B is the desired product)
# rate1 = k1 * C_A,  rate2 = k2 * C_B
# k_i = A_i * exp(-Ea_i / (R * T))

A1 = 1e6  # pre-exponential factor, reaction 1 [1/s]
A2 = 1e8  # pre-exponential factor, reaction 2 [1/s]
EA1 = 5e4  # activation energy, reaction 1 [J/mol]
EA2 = 7e4  # activation energy, reaction 2 [J/mol]
R_GAS = 8.314  # universal gas constant [J/(mol·K)]


def cstr_steady_state(
    temperature: float,
    residence_time: float,
    inlet_concentration: float,
    coolant_flow: float,
) -> dict[str, float]:
    """Compute steady-state CSTR outputs.

    Args:
        temperature: Reactor temperature [K].
        residence_time: Mean residence time [s].
        inlet_concentration: Inlet concentration of A [mol/L].
        coolant_flow: Coolant flow rate [L/s] (affects energy cost).

    Returns:
        Dictionary with conversion, selectivity, and energy_cost.
    """
    k1 = A1 * math.exp(-EA1 / (R_GAS * temperature))
    k2 = A2 * math.exp(-EA2 / (R_GAS * temperature))

    tau = residence_time
    ca = inlet_concentration / (1.0 + k1 * tau)
    cb = (k1 * tau * inlet_concentration) / ((1.0 + k1 * tau) * (1.0 + k2 * tau))

    conversion = 1.0 - ca / inlet_concentration
    selectivity = cb / (inlet_concentration - ca) if conversion > 1e-12 else 0.0

    # Energy cost: heating + coolant pumping (simplified)
    energy_cost = 0.01 * (temperature - 300.0) ** 2 + 5.0 * coolant_flow

    return {
        "conversion": conversion,
        "selectivity": selectivity,
        "energy_cost": energy_cost,
    }


# --8<-- [end:kinetics]


# ── Simulator and scorer ───────────────────────────────────────────


# --8<-- [start:world]
class CSTRSimulator:
    """Ground-truth CSTR simulator (Simulator protocol)."""

    def generate(self, config: dict[str, Any]) -> tuple[Any, Any]:
        """Generate ground truth and noisy observations.

        Args:
            config: Must contain temperature, residence_time,
                inlet_concentration, and coolant_flow.

        Returns:
            Tuple of (truth_dict, noisy_observations_dict).
        """
        truth = cstr_steady_state(
            temperature=config["temperature"],
            residence_time=config["residence_time"],
            inlet_concentration=config["inlet_concentration"],
            coolant_flow=config["coolant_flow"],
        )
        rng = np.random.default_rng(hash(frozenset(config.items())) % 2**32)
        noisy = {
            k: max(0.0, v + rng.normal(0, 0.02 * abs(v) + 1e-6))
            for k, v in truth.items()
        }
        return truth, noisy


class CSTRScorer:
    """Score CSTR outputs (Scorer protocol)."""

    def score(
        self,
        truth: Any,
        observations: Any,
        config: dict[str, Any],
    ) -> dict[str, float]:
        """Return observed values as scores.

        Args:
            truth: Ground truth dict (unused — we score observations).
            observations: Noisy observations dict.
            config: Trial configuration (unused).

        Returns:
            Scores for each observable.
        """
        return {
            "conversion": observations["conversion"],
            "selectivity": observations["selectivity"],
            "energy_cost": observations["energy_cost"],
        }


# --8<-- [end:world]


# ── Study definition ───────────────────────────────────────────────

# --8<-- [start:observables]
observables = [
    Observable("conversion", Direction.MAXIMIZE),
    Observable("selectivity", Direction.MAXIMIZE),
    Observable("energy_cost", Direction.MINIMIZE),
]
# --8<-- [end:observables]

# --8<-- [start:factors]
factors = [
    Factor("temperature", FactorType.CONTINUOUS, bounds=(320.0, 400.0)),
    Factor("residence_time", FactorType.CONTINUOUS, bounds=(10.0, 120.0)),
    Factor("inlet_concentration", FactorType.CONTINUOUS, bounds=(0.5, 3.0)),
    Factor("coolant_flow", FactorType.CONTINUOUS, bounds=(0.5, 5.0)),
]
# --8<-- [end:factors]

# --8<-- [start:phases]
# Phase 1: broad exploration with Latin hypercube
discovery_grid = build_grid(factors, method="lhs", n_samples=60, seed=42)

# Phase 2: refine the top 20 designs from Phase 1
phases = [
    Phase(
        name="discovery",
        grid=discovery_grid,
        filter_fn=top_k_pareto_filter(20),
    ),
    Phase(
        name="refinement",
        grid="carry",  # filled by carry-forward from Phase 1
    ),
]
# --8<-- [end:phases]


def _plot_kinetics(plt: Any) -> None:
    """Plot conversion & selectivity vs temperature for several residence times."""
    temps = np.linspace(320, 400, 200)
    tau_values = [20.0, 60.0, 100.0]
    fig, (ax_conv, ax_sel) = plt.subplots(1, 2, figsize=(10, 4))
    for tau in tau_values:
        conv = []
        sel = []
        for t_val in temps:
            out = cstr_steady_state(t_val, tau, 1.5, 2.0)
            conv.append(out["conversion"])
            sel.append(out["selectivity"])
        label = f"τ = {tau:.0f} s"
        ax_conv.plot(temps, conv, label=label)
        ax_sel.plot(temps, sel, label=label)
    ax_conv.set_xlabel("Temperature [K]")
    ax_conv.set_ylabel("Conversion")
    ax_conv.set_title("Conversion vs Temperature")
    ax_conv.legend()
    ax_sel.set_xlabel("Temperature [K]")
    ax_sel.set_ylabel("Selectivity")
    ax_sel.set_title("Selectivity vs Temperature")
    ax_sel.legend()
    fig.tight_layout()
    fig.savefig(f"{ASSET_DIR}/cstr_kinetics.png", dpi=150, bbox_inches="tight")
    print("\nSaved cstr_kinetics.png")
    plt.close(fig)


def main() -> None:
    """Run the CSTR trade study and print results."""
    # --8<-- [start:run]
    study = Study(
        world=CSTRSimulator(),
        scorer=CSTRScorer(),
        observables=observables,
        phases=phases,
    )
    study.run()
    # --8<-- [end:run]

    # --8<-- [start:results]
    # Phase 1 summary
    r1 = study.results("discovery")
    print(f"Discovery: {len(r1.configs)} trials")

    # Phase 2 summary
    r2 = study.results("refinement")
    print(f"Refinement: {len(r2.configs)} trials")

    # Pareto front
    front_idx = study.front("refinement")
    print(f"\nPareto front: {len(front_idx)} designs")
    print(f"{'Conv':>8s}  {'Select':>8s}  {'Energy':>8s}")
    for i in front_idx:
        c, s, e = r2.scores[i]
        print(f"{c:8.3f}  {s:8.3f}  {e:8.2f}")

    # Hypervolume (higher = better spread)
    ref = np.array([0.0, 0.0, 200.0])  # worst-case reference point
    hv = study.front_hypervolume("refinement", ref)
    print(f"\nHypervolume: {hv:.2f}")

    # Pareto front from extract_front (standalone usage)
    front_standalone = extract_front(
        r2.scores,
        [o.direction for o in observables],
    )
    print(f"Standalone extract_front: {len(front_standalone)} designs")
    # --8<-- [end:results]

    # --8<-- [start:plots]
    import matplotlib.pyplot as plt

    directions = [o.direction for o in observables]

    # ── Domain-specific: conversion & selectivity vs temperature ───
    _plot_kinetics(plt)

    # ── Trade-study plots ──────────────────────────────────────────
    # Pareto front scatter (3 objectives → pairwise matrix)
    fig_front, _ = plot_front(r2, directions)
    fig_front.savefig(f"{ASSET_DIR}/cstr_front.png", dpi=150, bbox_inches="tight")
    print("Saved cstr_front.png")
    plt.close(fig_front)

    # Parallel coordinates
    fig_par, _ = plot_parallel(r2, directions)
    fig_par.savefig(f"{ASSET_DIR}/cstr_parallel.png", dpi=150, bbox_inches="tight")
    print("Saved cstr_parallel.png")
    plt.close(fig_par)

    # Selectivity strip plot
    fig_sel2, _ = plot_scores(r2, "selectivity", directions)
    fig_sel2.savefig(
        f"{ASSET_DIR}/cstr_selectivity.png",
        dpi=150,
        bbox_inches="tight",
    )
    print("Saved cstr_selectivity.png")
    plt.close(fig_sel2)
    # --8<-- [end:plots]


if __name__ == "__main__":
    main()
