"""Run the actual README quickstart and compare its standalone workflow."""

from __future__ import annotations

import re
import runpy
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def test_readme_and_standalone_replicates_agree():
    """Both published entry points preserve pairing and the analytic cost."""
    blocks = re.findall(
        r"```python\n(.*?)\n```", (ROOT / "README.md").read_text(), re.DOTALL
    )
    namespace: dict[str, Any] = {}
    exec(blocks[0], namespace)  # ruff: ignore[exec-builtin] -- trusted code from the repository README
    results = namespace["results"]
    assert results.scores.shape == (128, 2)
    np.testing.assert_array_equal(namespace["means"].scores[:, 1], [8.0, 32.0])
    assert namespace["contrast"].n_pairs == 64
    assert namespace["contrast"].mean < 0.0
    assert sorted(namespace["front"].tolist()) == [0, 1]
    example = runpy.run_path(str(ROOT / "examples/replicated_study.py"))
    np.testing.assert_array_equal(example["run_example"]().scores, results.scores)


def test_bayesian_example_scores_joint_members_in_the_right_order():
    """A two-point, three-variable ensemble has analytic energy score 0.5."""
    example = runpy.run_path(str(ROOT / "examples/bayesian_study.py"))
    observations = {
        "pred_mean": np.array([1.0, 0.0, 0.0]),
        "pred_samples": np.array([[0.0, 2.0], [0.0, 0.0], [0.0, 0.0]]),
    }
    result = example["BayesianRegressionScorer"]().score(np.zeros(3), observations, {})
    # E||X-y|| = 1; E||X-X'|| = 1, hence ES = 1 - 1/2.
    assert np.isclose(result["energy"], 0.5)
    assert np.isclose(result["crps"], 1.0 / 6.0)
