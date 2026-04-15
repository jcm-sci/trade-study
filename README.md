# trade-study

[![CI](https://github.com/jcm-sci/trade-study/actions/workflows/ci.yml/badge.svg)](https://github.com/jcm-sci/trade-study/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/trade-study)](https://pypi.org/project/trade-study/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19599839.svg)](https://doi.org/10.5281/zenodo.19599839)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![jcm-sci](https://img.shields.io/badge/jcm--sci-jcmacdonald.dev-blue)](https://jcmacdonald.dev/software/)

Multi-objective trade-study orchestration: define factors, build
parameter grids, run hierarchical study phases, and extract Pareto
fronts — for any domain where you compare alternatives against
competing objectives.

## Statement of need

Comparing design alternatives against multiple objectives is a common
task across scientific simulation, engineering trade-offs, and ML
hyperparameter tuning. Researchers typically glue together separate
tools for grid construction, execution, scoring, and Pareto analysis,
writing ad-hoc scripts that are hard to reproduce or extend to
multi-phase studies (screening → refinement → benchmark).

`trade-study` provides a single orchestration layer that composes these
steps into a reproducible, protocol-driven workflow. Users supply a
`Simulator` (generates data) and a `Scorer` (evaluates it); the
framework handles grid construction, parallel execution, Pareto
extraction, and phase chaining. All components are modular and
optional — use only what you need.

This package targets researchers and practitioners who need:

- structured multi-phase experimental design (screen → refine → benchmark),
- multi-objective Pareto analysis across heterogeneous factors, and
- a reproducible Python API that separates domain logic from study orchestration.

## Why trade-study?

| Need | Without trade-study | With trade-study |
|------|---------------------|------------------|
| Parameter grid | Manual `itertools.product` or one-off scripts | `build_grid(factors, method="sobol")` — full factorial, LHS, Sobol, Halton |
| Multi-objective ranking | Call pymoo directly, handle direction normalization | `extract_front(scores, directions)` — direction-aware |
| Phased studies | Custom loop with manual filtering between stages | `Study(phases=[Phase(..., filter_fn=top_k_pareto_filter(k=20)), ...])` |
| Adaptive search | Set up optuna study from scratch | `run_adaptive(world, scorer, factors, observables, n_trials=600)` |
| Reproducibility | Scattered scripts, no standard protocol | `Simulator` / `Scorer` protocols + `save_results()` / `load_results()` |

Existing tools solve pieces of this problem — [optuna](https://optuna.org/) for adaptive optimization, [pymoo](https://pymoo.org/) for multi-objective solvers, [SALib](https://salib.readthedocs.io/) for sensitivity analysis — but none provide the **hierarchical phase orchestration** that connects them into a single study.

## Quick start

```python
from trade_study import (
    Direction,
    Factor,
    FactorType,
    Observable,
    Phase,
    Study,
    build_grid,
    top_k_pareto_filter,
)

# 1. Define objectives
accuracy = Observable("accuracy", Direction.MAXIMIZE)
latency = Observable("latency_ms", Direction.MINIMIZE)
cost = Observable("cost_usd", Direction.MINIMIZE)

# 2. Define factors and build a design grid
factors = [
    Factor("learning_rate", FactorType.CONTINUOUS, bounds=(1e-4, 1e-1)),
    Factor("backend", FactorType.CATEGORICAL, levels=["A", "B", "C"]),
]
grid = build_grid(factors, method="lhs", n_samples=500)

# 3. Run a hierarchical study
study = Study(
    world=MySimulator(),  # implements Simulator protocol
    scorer=MyScorer(),  # implements Scorer protocol
    observables=[accuracy, latency, cost],
    phases=[
        Phase(
            "screening",
            grid=grid,
            filter_fn=top_k_pareto_filter(k=20),
        ),
        Phase("benchmark", grid="carry", filter_fn=None),
    ],
)
study.run(n_jobs=-1)

# 4. Inspect results
print(study.summary())
front = study.front()  # non-dominated configs
hv = study.front_hypervolume(ref=...)  # hypervolume indicator
```

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

| Extra | Packages | Purpose |
|-------|----------|---------|
| `design` | [pyDOE3](https://github.com/relf/pyDOE3), [SALib](https://github.com/SALib/SALib), [scipy](https://scipy.org/) | Grid construction and sensitivity screening |
| `pareto` | [pymoo](https://pymoo.org/) | Non-dominated sorting and indicators |
| `scoring` | [scoringrules](https://github.com/frazane/scoringrules) | Proper scoring rules (CRPS, WIS, etc.) |
| `stacking` | [arviz](https://github.com/arviz-devs/arviz), scipy | Bayesian and score-based ensemble weights |
| `adaptive` | [optuna](https://optuna.org/) | Multi-objective Bayesian optimization |
| `parallel` | joblib | Parallel grid execution |
| `all` | All of the above | |

**Core dependency**: numpy only.

## API overview

### Design

```python
from trade_study import Factor, FactorType, build_grid, screen

factors = [
    Factor("x", FactorType.CONTINUOUS, bounds=(0.0, 1.0)),
    Factor("method", FactorType.CATEGORICAL, levels=["a", "b", "c"]),
]

grid = build_grid(factors, method="sobol", n_samples=1024)
si = screen(run_fn, factors, method="morris", n_eval=3000)
```

### Execution

```python
from trade_study import run_grid, run_adaptive

# Grid mode: evaluate all configs (optional parallelism)
results = run_grid(world, scorer, grid, observables, n_jobs=-1)

# Adaptive mode: multi-objective Bayesian optimization (NSGA-II)
results = run_adaptive(world, scorer, factors, observables, n_trials=600)
```

### Pareto analysis

```python
from trade_study import extract_front, hypervolume, pareto_rank

front_mask = extract_front(results.scores, directions)
ranks = pareto_rank(results.scores, directions)
hv = hypervolume(results.scores[front_mask], directions, ref=ref_point)
```

### Stacking

```python
from trade_study import stack_scores, stack_bayesian, ensemble_predict

weights = stack_scores(score_matrix)  # simplex-constrained optimization
weights = stack_bayesian(idata_dict)  # arviz stacking (Yao et al. 2018)
combined = ensemble_predict(predictions, weights)
```

### I/O

```python
from trade_study import save_results, load_results

save_results(results, "study_results")
results = load_results("study_results")
```

## Related packages

| Package | Description |
|---------|-------------|
| [TradeStudy.jl](https://github.com/jcm-sci/TradeStudy.jl) | Julia implementation of the same framework |

## Development

```bash
uv sync --extra dev
just ci          # lint → mypy --strict → pytest with coverage
just format      # auto-format
just check       # auto-fix lint
```

## License

MIT
