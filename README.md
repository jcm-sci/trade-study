# trade-study

[![CI](https://github.com/jcm-sci/trade-study/actions/workflows/ci.yml/badge.svg)](https://github.com/jcm-sci/trade-study/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/trade-study)](https://pypi.org/project/trade-study/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19599838.svg)](https://doi.org/10.5281/zenodo.19599838)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![jcm-sci](https://img.shields.io/badge/jcm--sci-jcmacdonald.dev-blue)](https://jcmacdonald.dev/projects/)

`trade-study` compares simulation configurations against multiple objectives.
Define factors, build parameter grids, run replicated study phases, and
extract Pareto fronts using a simulator and scorer supplied by the caller.

## Statement of need

A study may need grids, simulation runs, scoring, and comparisons across
several phases. `trade-study` represents these steps with explicit factors,
objectives, and phase filters. Users supply a `Simulator` to generate data
and a `Scorer` to evaluate it. The package handles grid construction,
replicated execution, Pareto extraction, and phase chaining. Optional
dependencies add parallel execution, sampled designs, and other integrations.

This package targets researchers and practitioners who need:

- structured multi-phase experimental design (screen → refine → benchmark),
- multi-objective Pareto analysis across heterogeneous factors, and
- a reproducible Python API that separates domain logic from study orchestration.

## Study operations

| Operation | Interface |
| --- | --- |
| Parameter grid | `build_grid` with full factorial, LHS, Sobol, or Halton designs |
| Multi-objective ranking | `extract_front(scores, directions)` |
| Phased studies | `Study`, `Phase`, and phase filters such as `top_k_pareto_filter` |
| Adaptive search | `run_adaptive` with an Optuna integration |
| Repeated evaluation and storage | `Simulator` and `Scorer` protocols, `save_results`, and `load_results` |

Optional integrations use [Optuna](https://optuna.org/) for adaptive search,
[pymoo](https://pymoo.org/) for Pareto algorithms, and
[SALib](https://salib.readthedocs.io/) for sensitivity analysis.
`Study` and `Phase` connect these operations across stages.

## Quick start

This complete example needs NumPy and the `pareto` extra:
`pip install 'trade-study[pareto]'`. Each replicate uses the same sample bank
for both designs, so errors can be compared in pairs.

```python
import numpy as np

from trade_study import (
    Direction, Factor, FactorType, Observable, build_grid,
    extract_front, paired_difference, run_grid,
)


class SampleMean:
    def generate(self, config, *, rep=0):
        rng = np.random.default_rng(np.random.SeedSequence([2026, rep]))
        return 0.0, rng.normal(size=32)[:config["samples"]]


class ErrorAndCost:
    def score(self, truth, observed, config):
        return {"error": float((observed.mean() - truth)**2),
                "cost": float(config["samples"])}


grid = build_grid(
    [Factor("samples", FactorType.DISCRETE, levels=[8, 32])], method="full"
)
observables = [Observable("error", Direction.MINIMIZE),
               Observable("cost", Direction.MINIMIZE)]
results = run_grid(SampleMean(), ErrorAndCost(), grid, observables, n_reps=64)
means = results.aggregate_replicates()
front = extract_front(means.scores, [o.direction for o in observables])
contrast = paired_difference(results, {"samples": 32}, {"samples": 8},
                             "error", seed=2026)
print(means.scores, front, contrast.mean, contrast.n_pairs)
```

`results` retains all 128 trial rows; `means` has one row per design. Bootstrap
resamples in the paired interval do not increase the number of independent
replicates. Inspect `n_pairs`: this helper excludes nonfinite pairs. A study
must define its failure policy before interpreting a success-only comparison.
Positive objective scaling preserves Pareto membership; use decision preferences
when choosing among nondominated designs.

The typed standalone version is [replicated_study.py](examples/replicated_study.py).
The [capabilities and example checks](docs/guide/capabilities.md) map additional
APIs, optional dependencies, and executable examples.

For a complete 15-minute demonstration with live code and saved figures, see
the [serosurvey design notebook](examples/serosurvey_study.ipynb) and
[presentation guide](https://jcm-sci.github.io/trade-study/guide/serosurvey/).
From a repository checkout, launch it with
`uv run --extra notebook jupyter lab examples/serosurvey_study.ipynb`.

### Protocols

Users implement two protocols to plug in their domain:

```python
from trade_study import Scorer, Simulator


class MySimulator:
    """Implements the Simulator protocol."""

    def generate(self, config: dict) -> tuple:
        """Return (truth, observations) for a given config."""
        ...


class MyScorer:
    """Implements the Scorer protocol."""

    def score(self, truth, observations, config: dict) -> dict[str, float]:
        """Return {observable_name: value} for a single trial."""
        ...
```

## Installation

```bash
pip install trade-study[all]
```

Or install only the extras you need:

```bash
pip install trade-study[design,pareto]
```

| Extra      | Packages                                                                                                       | Purpose                                     |
| ---------- | -------------------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| `design`   | [pyDOE3](https://github.com/relf/pyDOE3), [SALib](https://github.com/SALib/SALib), [scipy](https://scipy.org/) | Grid construction and sensitivity screening |
| `pareto`   | [pymoo](https://pymoo.org/)                                                                                    | Non-dominated sorting and indicators        |
| `scoring`  | [scoringrules](https://github.com/frazane/scoringrules)                                                        | Proper scoring rules (CRPS, WIS, etc.)      |
| `stacking` | [arviz](https://github.com/arviz-devs/arviz), scipy                                                            | Bayesian and score-based ensemble weights   |
| `adaptive` | [optuna](https://optuna.org/)                                                                                  | Adaptive multi-objective search (NSGA-II)   |
| `parallel` | joblib                                                                                                         | Parallel grid execution                     |
| `viz` | matplotlib | Plot scores and Pareto fronts |
| `surrogate` | scikit-learn | GP/RF score and regime surrogates |
| `dataframe` | pandas | ResultsTable export for analysis and CSV |
| `all`      | All of the above                                                                                               |                                             |
| `notebook` | JupyterLab, nbconvert, matplotlib, pandas, pymoo | Execute and present example notebooks |

**Core dependency**: numpy only.

## API overview

The snippets below show API composition; names such as `world`, `scorer`,
`run_fn` and `results` stand for caller-supplied objects. Use the complete
quickstart above or a worked guide to run a study.

### Design

```python
from trade_study import Factor, FactorType, build_grid, screen

factors = [
    Factor("x", FactorType.CONTINUOUS, bounds=(0.0, 1.0)),
    Factor("method", FactorType.CATEGORICAL, levels=["a", "b", "c"]),
]

grid = build_grid(factors, method="sobol", n_samples=1024)
si = screen(run_fn, factors, method="morris", n_trajectories=3000)
```

### Execution

```python
from trade_study import run_grid, run_adaptive

# Grid mode: evaluate all configs (optional parallelism)
results = run_grid(world, scorer, grid, observables, n_jobs=-1)

# Adaptive mode: evolutionary multi-objective optimization (Optuna NSGA-II)
results = run_adaptive(world, scorer, factors, observables, n_trials=600)
```

### Pareto analysis

```python
from trade_study import extract_front, hypervolume, pareto_rank

front_idx = extract_front(results.scores, directions)
ranks = pareto_rank(results.scores, directions)
hv = hypervolume(results.scores[front_idx], ref_point, directions)
```

### Stacking

```python
from trade_study import stack_scores, stack_bayesian, stack_proportional, ensemble_predict

weights = stack_scores(score_matrix)  # select best means; split exact ties
weights = stack_proportional(score_matrix)  # preserve weight across near ties
weights = stack_bayesian(idata_dict)  # arviz stacking (Yao et al. 2018)
combined = ensemble_predict(predictions, weights)
```

`stack_scores` solves the supplied linear-score objective directly using NumPy;
positive changes of score units preserve its optimum. `ensemble_predict`
normalizes a copy of the weights and accepts read-only arrays and views.
See [choosing ensemble weights](docs/guide/stacking.md) for examples and the
distinction between input scores and the score of a combined prediction.

### I/O

```python
from trade_study import save_results, load_results

save_results(results, "study_results")
results = load_results("study_results")
frame = results.to_dataframe()  # optional pandas extra; includes trial metadata
frame.to_csv("study_results.csv", index=False)
```

### Multi-fidelity search and surrogates

```python
from trade_study import run_successive_halving, run_hyperband, fit_surrogate, fit_regime_surrogate

# Cheap-first budget-constrained screening
results = run_successive_halving(world, scorer, factors, observables, budget_param="maxiters")
results = run_hyperband(world, scorer, factors, observables, budget_param="maxiters")

# Interpolate scores from a completed results table (GP or RF)
surrogate = fit_surrogate(results, method="gp")

# Recommend a config per regime, interpolating over regime descriptors
regime_surrogate = fit_regime_surrogate(results, regime_cols=["n", "p"])
```

Surrogate fits report cross-validated `cv_r2` and `cv_rmse`; check those
diagnostics before trusting recommendations or post-hoc sensitivity results.

`Constraint` / `feasibility_filter` express infeasible regions of the
design space; `FactorConstraint` couples factors during grid
construction (e.g. keep `x + y <= 1`); `screen()` supports both Morris
and Sobol sensitivity analysis. `run_grid(..., n_reps=N)` supports replicated
trials, and `sensitivity_from_table()` estimates post-hoc sensitivity from an
existing results table through a validated surrogate.

## Release compatibility

Version 0.5.1 fixes score-unit sensitivity in `stack_scores` and prevents
`ensemble_predict` from modifying supplied weights, including read-only arrays
and views. It also updates executable documentation and illustrated examples.

Version 0.5.0 adds a prespecified fixed replication policy and versioned
replication-contract metadata. Adaptive allocation remains the default.
Pareto operations now identify the optional `pareto` extra when it is missing.

Version 0.4.0 adds decision reporting (`preference_sweep()` with regret,
practical equivalence and optional paired uncertainty), fixed-sample proportion
intervals and replicate sizing, versioned result bundles with provenance,
sequential replication (`run_sequential()`, with explicit fixed/adaptive policy
and recorded counts/uncertainty assumptions), evaluation caching, incremental
grid checkpoints with bounded retries, grouped surrogate validation with
support diagnostics, and the IVAC serosurvey notebook. The `adaptive` extra now
requires Optuna 4.5 or later, and strict `<`/`>` adaptive constraints treat
exact equality as infeasible, matching result-table feasibility.

Version 0.3.0 adds replicated and paired studies, uncertainty diagnostics,
persistent adaptive sessions, checkpointing, and DataFrame export. Existing
phase-name-only checkpoints are refused; use a fresh checkpoint directory and
set `checkpoint_key` to track model/data revisions. Confidence constraints
require estimated standard errors rather than assuming unknown uncertainty is
zero. `stack_proportional()` now uses exponential score utilities in both
directions; set its temperature in score units to control concentration.

## Related packages

| Package                                                   | Description                                                                                     |
| --------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| [TradeStudy.jl](https://github.com/jcm-sci/TradeStudy.jl) | Inactive design scaffold for a possible Julia port; it does not contain a usable implementation |

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, code standards, and the PR workflow.

```bash
uv sync --extra dev
just ci          # lint → mypy --strict → pytest with coverage
just format      # auto-format
just check       # auto-fix lint
```

## Citation

If you use this package in your research, please cite:

```bibtex
@software{trade_study2026,
  author = {Macdonald, Joshua C.},
  title = {{trade-study}: Multi-Objective Trade-Study Orchestration},
  year = {2026},
  url = {https://github.com/jcm-sci/trade-study},
  version = {0.5.1},
  doi = {10.5281/zenodo.19599838},
}
```

See [CITATION.cff](CITATION.cff) for machine-readable metadata.

## License

MIT
