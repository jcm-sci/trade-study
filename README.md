# trade-study

[![jcm-sci](https://img.shields.io/badge/jcm--sci-jcmacdonald.dev-blue)](https://jcmacdonald.dev/software/)

Multi-objective trade-study orchestration: scoring, Pareto optimization,
and Bayesian stacking for scientific model evaluation.

## Overview

`trade-study` provides a structured framework for evaluating scientific
simulation models against known ground truth via observable-based scoring,
multi-objective Pareto optimization, and Bayesian model stacking.

The core pattern (from [MFAI §4–5](https://github.com/jcm-sci/trade-study)):

1. **Model world** — Generate synthetic ground truth with known latent state
2. **Observe** — Apply a realistic observation model (noise, masks, bias)
3. **Score** — Evaluate structured observables against *known truth* using
   proper scoring rules (not noisy observations)
4. **Pareto** — Extract the multi-objective Pareto front across competing
   objectives (e.g. quality vs. cost)
5. **Stack** — Combine models via Bayesian stacking weights or score-based
   optimization

Studies can be organized into **hierarchical phases** (discovery → refinement →
benchmark), where each phase filters configs for the next.

## Architecture

```
trade_study/
├── protocols.py    ModelWorld, Scorer, Observable, Annotation, ResultsTable
├── design.py       Factor screening (SALib) and grid construction (pyDOE3)
├── scoring.py      Proper scoring rules (scoringrules) and calibration
├── pareto.py       Non-dominated sorting and indicators (pymoo)
├── stacking.py     Bayesian stacking (arviz) and score-based weights (scipy)
├── runner.py       Grid execution (joblib) and adaptive search (optuna)
├── study.py        Multi-phase orchestration with filtering
└── io.py           Save/load results (npz + JSON)
```

### Observable Hierarchy (MFAI §4)

| Tier | Role | Examples |
|------|------|---------|
| **Embedded** | Must hold by construction | ELBO monotonicity, conservation laws |
| **Penalized** | Soft constraints in objective | Stopping criteria, convergence rate |
| **Diagnostic** | Post-hoc evaluation only | Coverage, CRPS, PIT, rank error |
| **Cost** | Resource axis for Pareto | Wall time, dollar cost, field teams |

### Two Execution Modes

- **Grid mode**: Full factorial, LHS, or fractional factorial via pyDOE3 →
  run all → post-hoc Pareto extraction via pymoo.
- **Adaptive mode**: Multi-objective Bayesian optimization via optuna NSGA-II
  when the full grid is too expensive.

### Stacking: Two Paths

- **Bayesian** (arviz): For models with log-likelihoods. Implements Yao et al
  2018 stacking via `arviz.compare(method='stacking')`.
- **Score-based** (scipy): For arbitrary score matrices. Optimizes simplex
  weights to minimize/maximize composite score.

## Quick Example

```python
from trade_study import Study, Phase, Observable, Tier, Direction
from trade_study.study import top_k_pareto_filter
from trade_study.design import build_grid, Factor, FactorType

# Define observables
coverage = Observable("coverage_95", Tier.DIAGNOSTIC, Direction.MAXIMIZE)
rank_err = Observable("rank_error", Tier.EMBEDDED, Direction.MINIMIZE)
wall_time = Observable("wall_seconds", Tier.COST, Direction.MINIMIZE)

# Build design grid
factors = [
    Factor("alpha", FactorType.CONTINUOUS, bounds=(0.01, 0.10)),
    Factor("method", FactorType.CATEGORICAL, levels=["A", "B", "C"]),
]
grid = build_grid(factors, method="lhs", n_samples=500)

# Run hierarchical study
study = Study(
    world=MyModelWorld(),
    scorer=MyScorer(),
    observables=[coverage, rank_err, wall_time],
    phases=[
        Phase("discovery", grid=grid, filter_fn=top_k_pareto_filter(k=20)),
        Phase(
            "benchmark",
            grid="carry",  # top-20 from discovery
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
| `design` | [pyDOE3](https://github.com/relf/pyDOE3), [SALib](https://github.com/SALib/SALib) |
| `adaptive` | [optuna](https://optuna.org/) |
| `parallel` | joblib |
| `all` | All of the above |

## Consumers

| Package | Domain | Use Case |
|---------|--------|----------|
| [VBPCApy](https://github.com/yoavram-lab/VBPCApy) | Bayesian PCA | Convergence sweep: 15-factor grid, coverage/RMSE/CRPS vs. wall time |
| [pp-eigentest](https://github.com/yoavram-lab/pp-eigentest) | Rank selection | 3-stage method selection: type-I/power/exact vs. robustness |
| TICCS | Surveillance design | relWIS/peak-timing vs. surveillance cost ($) |

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
