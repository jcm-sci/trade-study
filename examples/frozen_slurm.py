"""Run a frozen, packed Gaussian study locally; submit only with --submit.

The generated project is a small committed Python module, independent of the
submitting checkout. No cluster account or external scientific data is needed.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess  # ruff: ignore[suspicious-subprocess-import] - explicit Git argv
from pathlib import Path

from trade_study.slurm import (
    SlurmResources,
    SlurmTask,
    freeze_checkout,
    prepare_slurm_plan,
)

_PROJECT = '''"""A normal-mean comparison with one grid ledger per launch task."""
import json
import sys
from pathlib import Path
import numpy as np
from trade_study import Direction, Observable, run_grid

class World:
    def generate(self, config, *, rep=0):
        rng = np.random.default_rng(np.random.SeedSequence([2026, rep]))
        return 0.0, rng.normal(size=32)[:config["n"]].mean()

class Scorer:
    def score(self, truth, observation, config):
        return {"squared_error": float((truth - observation)**2)}

def main():
    root, index = Path(sys.argv[1]), int(sys.argv[2])
    root.mkdir(parents=True, exist_ok=True)
    n = (8, 32)[index]
    result = run_grid(World(), Scorer(), [{"n": n}],
        [Observable("squared_error", Direction.MINIMIZE)], n_reps=16,
        checkpoint_path=root / f"grid-{index}.sqlite", checkpoint_key="normal-v1")
    (root / f"result-{index}.json").write_text(json.dumps({
        "n": n, "replicates": len(result.configs),
        "mean_squared_error": float(result.scores.mean()),
        "analytic_expected_error": 1.0 / n,
    }))

if __name__ == "__main__":
    main()
'''


def main() -> None:
    """Prepare the demonstration project and print its executed table or job ID.

    Raises:
        ValueError: An existing example directory has different source.
        FileNotFoundError: Git is unavailable.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    root = args.directory.resolve()
    source = root / "source"
    source.mkdir(parents=True, exist_ok=True)
    driver = source / "normal_study.py"
    if driver.exists() and driver.read_text() != _PROJECT:
        msg = "Existing example source differs; use a new directory"
        raise ValueError(msg)
    if not driver.exists():
        driver.write_text(_PROJECT)
        git = shutil.which("git")
        if git is None:
            msg = "Install Git for this frozen-checkout example"
            raise FileNotFoundError(msg)
        for command in (
            ("init", "-q"),
            ("add", "normal_study.py"),
            (
                "-c",
                "user.name=Study example",
                "-c",
                "user.email=example@example.org",
                "commit",
                "-qm",
                "Fixed analytic normal study",
            ),
        ):
            subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
                [git, "-C", str(source), *command], check=True, capture_output=True
            )
    frozen = freeze_checkout(source, root / "frozen")
    outputs = root / "results"
    plan = prepare_slurm_plan(
        frozen,
        [
            SlurmTask(
                ("-m", "normal_study", str(outputs), str(index)),
                outputs=(str(outputs / f"result-{index}.json"),),
            )
            for index in range(2)
        ],
        directory=root / "plan",
        revision="fixed-normal-bank-v1",
        imports=("normal_study",),
        resources=SlurmResources(width=2, cpus_per_task=1),
    )
    if args.submit:
        print(plan.submit())
        return
    plan.run_local()
    print("| n | Replicates | Empirical squared error | Analytic expected error |")
    print("| ---: | ---: | ---: | ---: |")
    for index in range(2):
        result = json.loads((outputs / f"result-{index}.json").read_text())
        print(
            f"| {result['n']} | {result['replicates']} | "
            f"{result['mean_squared_error']:.5f} | "
            f"{result['analytic_expected_error']:.5f} |"
        )


if __name__ == "__main__":
    main()
