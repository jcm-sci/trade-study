# model-criticism

Observable-based model evaluation, Pareto optimization, and Bayesian stacking
for scientific model criticism.

## Overview

`model-criticism` provides a structured framework for evaluating scientific
simulation models against data via observable-based scoring, multi-objective
Pareto optimization, and Bayesian model stacking.

The core pattern:

1. **Model world** — Run a simulation under a given configuration
2. **Observe** — Extract structured observables from the output
3. **Score** — Evaluate each observable against reference data using proper
   scoring rules
4. **Optimize** — Find Pareto-optimal configurations across multiple objectives
5. **Stack** — Combine models via Bayesian stacking weights

## Status

**Pre-alpha.** API is being designed. See
[VBPCApy convergence_design_matrix.md](https://github.com/yoavram-lab/VBPCApy)
for the motivating design document.

## Planned Dependencies

- Core: `numpy`, `scipy`
- Scoring: [`scoringrules`](https://github.com/frazane/scoringrules)
- Stacking: [`arviz`](https://github.com/arviz-devs/arviz)
- Pareto: [`pymoo`](https://pymoo.org/)
- Tuning: [`optuna`](https://optuna.org/)

## Related Packages

| Package | Description |
|---------|-------------|
| [ModelCriticism.jl](https://github.com/jcm-sci/ModelCriticism.jl) | Julia implementation of this same framework |
| [VBPCApy](https://github.com/yoavram-lab/VBPCApy) | First consumer — convergence sweep evaluation |
| [pp-eigentest](https://github.com/yoavram-lab/pp-eigentest) | First consumer — rank selection evaluation |

## Installation

```bash
pip install model-criticism
```

## Development

```bash
uv sync --extra dev
just ci
```

## License

MIT
