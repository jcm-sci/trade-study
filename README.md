# trade-study

[![jcm-sci](https://img.shields.io/badge/jcm--sci-jcmacdonald.dev-blue)](https://jcmacdonald.dev/software/)

Multi-objective design-of-experiments framework: generate parameter
grids, score competing configurations against multiple objectives,
extract Pareto fronts, and combine results via stacking — for any
domain where you compare alternatives.

## Overview

`trade-study` is a domain-agnostic framework for multi-objective
configuration comparison.  It works anywhere you need to evaluate
design alternatives against several objectives at once — scientific
simulation, ML hyperparameter tuning, engineering trade-offs, or
business decision analysis.

The core workflow:

1. **Design** — Build a parameter grid (full factorial, LHS, Sobol, Halton)
   or let an adaptive optimizer propose configs
2. **Evaluate** — Run each configuration through a user-supplied
   `Simulator` / `Scorer` protocol pair
3. **Score** — Compute per-config metrics (proper scoring rules, custom
   losses, KPIs — whatever your domain requires)
4. **Pareto** — Extract the non-dominated front across competing
   objectives (e.g. quality vs. cost)
5. **Stack** — Combine configurations via score-based or Bayesian
   stacking weights

Studies can be organized into **hierarchical phases** (screening →
refinement → benchmark), where each phase filters configs for the next.

## Architecture

```
trade_study/
├── protocols.py    Simulator & Scorer protocols, Observable, ResultsTable
├── design.py       Factor screening (SALib) and grid construction (pyDOE3/scipy)
├── scoring.py      Proper scoring rules (scoringrules) and common metrics
├── pareto.py       Non-dominated sorting and indicators (pymoo)
├── stacking.py     Score-based weights (scipy) and Bayesian stacking (arviz)
├── runner.py       Grid execution (joblib) and adaptive search (optuna)
├── study.py        Multi-phase orchestration with filtering
└── io.py           Save/load results (npz + JSON)
```

### Observable Hierarchy

| Tier | Role | Examples |
|------|------|---------|
| **Embedded** | Must hold by construction | Monotonicity, conservation laws, schema validity |
| **Penalized** | Soft constraints in objective | Convergence rate, latency budget, error threshold |
| **Diagnostic** | Post-hoc evaluation only | RMSE, coverage, CRPS, F1, rank error |
| **Cost** | Resource axis for Pareto | Wall time, dollar cost, memory, API calls |

### Two Execution Modes

- **Grid mode**: Full factorial, LHS, Sobol, or Halton via pyDOE3/scipy →
  run all → post-hoc Pareto extraction via pymoo.
- **Adaptive mode**: Multi-objective Bayesian optimization via optuna NSGA-II
  when the full grid is too expensive.

### Stacking: Two Paths

- **Score-based** (scipy): For arbitrary score matrices. Optimizes simplex
  weights to minimize/maximize composite score. Works with any metric.
- **Bayesian** (arviz): For models with log-likelihoods. Implements Yao et al.
  2018 stacking via `arviz.compare(method='stacking')`.

## Quick Example

```python
from trade_study import Study, Phase, Observable, Tier, Direction
from trade_study.study import top_k_pareto_filter
from trade_study.design import build_grid, Factor, FactorType

# Define objectives
accuracy = Observable("accuracy", Tier.DIAGNOSTIC, Direction.MAXIMIZE)
latency = Observable("latency_ms", Tier.PENALIZED, Direction.MINIMIZE)
cost = Observable("cost_usd", Tier.COST, Direction.MINIMIZE)

# Build design grid
factors = [
    Factor("learning_rate", FactorType.CONTINUOUS, bounds=(1e-4, 1e-1)),
    Factor("backend", FactorType.CATEGORICAL, levels=["A", "B", "C"]),
]
grid = build_grid(factors, method="lhs", n_samples=500)

# Run hierarchical study
study = Study(
    world=MySimulator(),   # implements Simulator protocol
    scorer=MyScorer(),     # implements Scorer protocol
    observables=[accuracy, latency, cost],
    phases=[
        Phase("screening", grid=grid, filter_fn=top_k_pareto_filter(k=20)),
        Phase(
            "benchmark",
            grid="carry",  # top-20 from screening
            filter_fn=None,
        ),
    ],
)
study.run(n_jobs=-1)
print(study.summary())
```

## Installation

```bash
pip install trade-study[all]
```

Or install only the extras you need:

```bash
pip install trade-study[scoring,pareto]  # just scoring + Pareto
```

| Extra | Packages |
|-------|----------|
| `scoring` | [scoringrules](https://github.com/frazane/scoringrules) |
| `pareto` | [pymoo](https://pymoo.org/) |
| `stacking` | [arviz](https://github.com/arviz-devs/arviz), scipy |
| `design` | [pyDOE3](https://github.com/relf/pyDOE3), [SALib](https://github.com/SALib/SALib), [scipy](https://scipy.org/) |
| `adaptive` | [optuna](https://optuna.org/) |
| `parallel` | joblib |
| `all` | All of the above |

## Consumers

| Package | Domain | Use Case |
|---------|--------|----------|
| [VBPCApy](https://github.com/yoavram-lab/VBPCApy) | Bayesian PCA | Convergence sweep: 13-factor grid, coverage/RMSE/CRPS vs. wall time |
| [pp-eigentest](https://github.com/yoavram-lab/pp-eigentest) | Rank selection | 3-stage method selection: type-I/power/exact vs. robustness |

## Related Packages

| Package | Description |
|---------|-------------|
| [TradeStudy.jl](https://github.com/jcm-sci/TradeStudy.jl) | Julia implementation of this same framework |

## Development

```bash
uv sync --extra dev
just ci
```

## License

MIT
