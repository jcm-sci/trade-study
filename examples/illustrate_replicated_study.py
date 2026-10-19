"""Plot the executable paired study against its analytic mean-squared error.

Regenerate with Python 3.12: uv run --extra dev --with matplotlib==3.11.2
python examples/illustrate_replicated_study.py. Use --check without plotting.
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from illustration_artifacts import IllustrationData, build_artifacts
from replicated_study import run_example

from trade_study import paired_difference

if TYPE_CHECKING:
    from collections.abc import Sequence


def _render(
    output: Path, samples: list[int], means: list[float], errors: list[float]
) -> None:
    mpl = importlib.import_module("matplotlib")
    mpl.use("Agg")
    plt = importlib.import_module("matplotlib.pyplot")
    plt.rcParams.update({"svg.hashsalt": "trade-study-replicates", "font.size": 10})
    figure, axis = plt.subplots(figsize=(6.5, 3.6), layout="constrained")
    reference = np.linspace(8, 32, 100)
    axis.plot(
        reference,
        1 / reference,
        color="#536774",
        linestyle="--",
        label="Analytic mean squared error: 1 / n",
    )
    axis.errorbar(
        samples,
        means,
        yerr=errors,
        color="#0d6b6e",
        fmt="o",
        capsize=5,
        label="64 replicates: mean +/- 1 Monte Carlo SE",
    )
    axis.set(
        xlabel="Sample cost n",
        ylabel="Squared-error objective (minimize)",
        title="Precision-cost tradeoff with paired replicate banks",
        xticks=samples,
    )
    axis.legend(frameon=False, fontsize=9)
    axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(output / "figure.svg", metadata={"Date": None})
    figure.savefig(
        output / "figure.pdf", metadata={"CreationDate": None, "ModDate": None}
    )
    plt.close(figure)


def _main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "docs/assets/replicated-study",
    )
    args = parser.parse_args(argv)
    results = run_example()
    samples = [8, 32]
    means: list[float] = []
    errors: list[float] = []
    lines = [
        (
            "| Samples / cost | Replicates | Mean squared error | "
            "Monte Carlo SE | Analytic mean (1 / n) |"
        ),
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for n in samples:
        values = results.scores[[c["samples"] == n for c in results.configs], 0]
        mean = float(values.mean())
        error = float(values.std(ddof=1) / np.sqrt(values.size))
        means.append(mean)
        errors.append(error)
        lines.append(
            f"| {n} | {values.size} | {mean:.5f} | {error:.5f} | {1 / n:.5f} |"
        )
    contrast = paired_difference(
        results, {"samples": 32}, {"samples": 8}, "error", seed=2026
    )
    lines.extend([
        "",
        (
            f"Paired mean difference (32 minus 8): **{contrast.mean:.5f}**, "
            f"using **{contrast.n_pairs}** finite pairs. "
            f"Analytic expected difference: **{1 / 32 - 1 / 8:.5f}**."
        ),
    ])
    build_artifacts(
        args.output,
        IllustrationData(
            "\n".join(lines) + "\n",
            {"samples": samples, "mean_error": means, "monte_carlo_se": errors},
        ),
        [Path(__file__), Path(__file__).with_name("replicated_study.py")],
        lambda output: _render(output, samples, means, errors),
        check=args.check,
    )


if __name__ == "__main__":
    _main()
