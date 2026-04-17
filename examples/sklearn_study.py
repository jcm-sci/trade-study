"""Scikit-learn hyperparameter trade study.

A multi-objective sweep over GradientBoostingRegressor hyperparameters
using the Friedman #1 synthetic dataset.  Objectives: RMSE (minimize),
training time (minimize), and model complexity (minimize).
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from sklearn.datasets import make_friedman1  # type: ignore[import-untyped]
from sklearn.ensemble import (  # type: ignore[import-untyped]
    GradientBoostingRegressor,
)
from sklearn.metrics import root_mean_squared_error  # type: ignore[import-untyped]
from sklearn.model_selection import train_test_split  # type: ignore[import-untyped]

from trade_study import (
    Direction,
    Factor,
    FactorType,
    Observable,
    build_grid,
    extract_front,
    plot_front,
    plot_parallel,
    plot_scores,
    run_grid,
)

ASSET_DIR = "docs/assets"

# ── Dataset ────────────────────────────────────────────────────────

# --8<-- [start:dataset]
X, y = make_friedman1(n_samples=800, noise=1.0, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
)
# --8<-- [end:dataset]


# ── Simulator and scorer ──────────────────────────────────────────


# --8<-- [start:world]
class GBSimulator:
    """Simulator that trains a GradientBoostingRegressor.

    The 'truth' is the test-set ground truth; 'observations' are the
    model's test-set predictions plus training metadata.
    """

    def generate(self, config: dict[str, Any]) -> tuple[Any, Any]:
        """Train a model and return predictions on the test set.

        Args:
            config: Hyperparameter dict with n_estimators, max_depth,
                learning_rate, and subsample.

        Returns:
            Tuple of (y_test array, observation dict with predictions,
            wall time, and number of leaves).
        """
        model = GradientBoostingRegressor(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            learning_rate=config["learning_rate"],
            subsample=config["subsample"],
            random_state=42,
        )
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        wall = time.perf_counter() - t0

        preds = model.predict(X_test)
        n_leaves = sum(tree[0].tree_.n_leaves for tree in model.estimators_)
        return y_test, {"predictions": preds, "wall": wall, "n_leaves": n_leaves}


class GBScorer:
    """Score GradientBoosting results for three objectives."""

    def score(
        self,
        truth: Any,
        observations: Any,
        config: dict[str, Any],
    ) -> dict[str, float]:
        """Compute RMSE, training time, and complexity.

        Args:
            truth: True target values (y_test).
            observations: Dict with predictions, wall time, n_leaves.
            config: Hyperparameter dict (unused).

        Returns:
            Scores for rmse, train_time, and complexity.
        """
        return {
            "rmse": root_mean_squared_error(truth, observations["predictions"]),
            "train_time": observations["wall"],
            "complexity": float(observations["n_leaves"]),
        }


# --8<-- [end:world]

# ── Study setup ────────────────────────────────────────────────────

# --8<-- [start:observables]
observables = [
    Observable("rmse", Direction.MINIMIZE),
    Observable("train_time", Direction.MINIMIZE),
    Observable("complexity", Direction.MINIMIZE),
]
# --8<-- [end:observables]

# --8<-- [start:factors]
factors = [
    Factor("n_estimators", FactorType.DISCRETE, levels=[50, 100, 200, 400]),
    Factor("max_depth", FactorType.DISCRETE, levels=[2, 3, 4, 5]),
    Factor("learning_rate", FactorType.DISCRETE, levels=[0.01, 0.05, 0.1, 0.2]),
    Factor("subsample", FactorType.DISCRETE, levels=[0.6, 0.8, 1.0]),
]
# --8<-- [end:factors]


def _plot_heatmap(plt: Any, results: Any) -> None:
    """Plot best RMSE heatmap over (n_estimators, max_depth) grid."""
    n_est_levels = sorted({c["n_estimators"] for c in results.configs})
    depth_levels = sorted({c["max_depth"] for c in results.configs})

    heat = np.full((len(depth_levels), len(n_est_levels)), np.nan)
    for idx_cfg, cfg in enumerate(results.configs):
        row = depth_levels.index(cfg["max_depth"])
        col = n_est_levels.index(cfg["n_estimators"])
        val = results.scores[idx_cfg, 0]
        heat[row, col] = (
            np.nanmin([heat[row, col], val]) if not np.isnan(heat[row, col]) else val
        )

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(heat, origin="lower", aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(n_est_levels)))
    ax.set_xticklabels(n_est_levels)
    ax.set_yticks(range(len(depth_levels)))
    ax.set_yticklabels(depth_levels)
    ax.set_xlabel("n_estimators")
    ax.set_ylabel("max_depth")
    ax.set_title("Best RMSE per (n_estimators, max_depth)")
    fig.colorbar(im, ax=ax, label="RMSE")
    fig.tight_layout()
    fig.savefig(f"{ASSET_DIR}/sklearn_heatmap.png", dpi=150, bbox_inches="tight")
    print("\nSaved sklearn_heatmap.png")
    plt.close(fig)


def main() -> None:
    """Run the hyperparameter trade study and print results."""
    # --8<-- [start:run]
    grid = build_grid(factors, method="full")
    print(f"Full factorial grid: {len(grid)} configurations")

    results = run_grid(
        world=GBSimulator(),
        scorer=GBScorer(),
        grid=grid,
        observables=observables,
    )
    # --8<-- [end:run]

    # --8<-- [start:results]
    # Pareto front
    front_idx = extract_front(
        results.scores,
        [o.direction for o in observables],
    )
    print(f"Pareto front: {len(front_idx)} / {len(grid)} designs\n")

    print(
        f"{'n_est':>6s}  {'depth':>5s}  {'lr':>6s}  {'sub':>5s}  "
        f"{'RMSE':>6s}  {'Time':>6s}  {'Leaves':>6s}"
    )
    print("-" * 52)
    for i in front_idx:
        cfg = results.configs[i]
        rmse, t, leaves = results.scores[i]
        print(
            f"{cfg['n_estimators']:6d}  {cfg['max_depth']:5d}  "
            f"{cfg['learning_rate']:6.2f}  {cfg['subsample']:5.1f}  "
            f"{rmse:6.3f}  {t:6.3f}  {leaves:6.0f}"
        )

    # Best RMSE on the front
    front_rmse = results.scores[front_idx, 0]
    best = front_idx[np.argmin(front_rmse)]
    print(f"\nLowest-RMSE Pareto design: {results.configs[best]}")
    print(
        f"  RMSE={results.scores[best, 0]:.4f}  "
        f"time={results.scores[best, 1]:.3f}s  "
        f"leaves={results.scores[best, 2]:.0f}"
    )
    # --8<-- [end:results]

    # --8<-- [start:plots]
    import matplotlib.pyplot as plt

    directions = [o.direction for o in observables]

    # ── Domain-specific: RMSE heatmap (n_estimators vs max_depth) ──
    _plot_heatmap(plt, results)

    # ── Trade-study plots ──────────────────────────────────────────
    # Pareto front scatter (3 objectives → pairwise matrix)
    fig_front, _ = plot_front(results, directions)
    fig_front.savefig(f"{ASSET_DIR}/sklearn_front.png", dpi=150, bbox_inches="tight")
    print("Saved sklearn_front.png")
    plt.close(fig_front)

    # Parallel coordinates
    fig_par, _ = plot_parallel(results, directions)
    fig_par.savefig(f"{ASSET_DIR}/sklearn_parallel.png", dpi=150, bbox_inches="tight")
    print("Saved sklearn_parallel.png")
    plt.close(fig_par)

    # RMSE strip plot
    fig_rmse, _ = plot_scores(results, "rmse", directions)
    fig_rmse.savefig(f"{ASSET_DIR}/sklearn_rmse.png", dpi=150, bbox_inches="tight")
    print("Saved sklearn_rmse.png")
    plt.close(fig_rmse)
    # --8<-- [end:plots]


if __name__ == "__main__":
    main()
