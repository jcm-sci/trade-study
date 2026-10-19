# trade-study

`trade-study` runs replicated simulation studies and compares configurations
against multiple objectives. It supports design grids, multi-fidelity phases,
Pareto analysis, response models, sensitivity analysis, and model weighting.

For installation and quick-start examples, see the
[README](https://github.com/jcm-sci/trade-study#readme).

For a short presentation with live code, tables and saved figures, see the
[serosurvey design notebook](guide/serosurvey.md), designed for a mixed IVAC audience.

## Overview

`trade-study` provides a structured workflow for multi-objective
design-of-experiments studies:

1. **Define** observables and design factors via lightweight
   [Protocols](api/protocols.md)
2. **Build** experimental grids with [Design](api/design.md) utilities
   (full factorial, Latin hypercube, Sobol, Halton, and constraints)
3. **Run** simulations across the grid, with optional replicates, adaptive
   NSGA-II search, or multi-fidelity allocation ([Runner](api/runner.md))
4. **Score** posterior predictive accuracy with proper scoring rules
   ([Scoring](api/scoring.md))
5. **Filter** the Pareto front ([Pareto](api/pareto.md))
6. **Stack** models via Bayesian stacking ([Stacking](api/stacking.md))
7. **Orchestrate** multi-phase studies with [Study](api/study.md)
8. **Fit and validate surrogates** for interpolation, regime-specific
   recommendations, and post-hoc sensitivity ([Surrogates](api/surrogate.md))

## API Reference

| Module | Description |
|--------|-------------|
| [Protocols](api/protocols.md) | Core types: `Observable`, `Direction`, `Scorer`, `Simulator`, etc. |
| [Design](api/design.md) | `Factor`, `FactorConstraint`, `build_grid`, `screen`, `sobol_indices`, `reduce_factors` |
| [Runner](api/runner.md) | `run_grid`, `run_adaptive`, `run_successive_halving`, `run_hyperband` |
| [Study](api/study.md) | `Phase`, `Study`, `top_k_pareto_filter`, `weighted_sum_filter`, `feasibility_filter` |
| [Scoring](api/scoring.md) | `score`, `coverage_curve` |
| [Pareto](api/pareto.md) | `extract_front`, `pareto_rank`, `hypervolume`, `igd_plus` |
| [Stacking](api/stacking.md) | `stack_bayesian`, `stack_scores`, `stack_proportional`, `ensemble_predict` |
| [Surrogates](api/surrogate.md) | `fit_surrogate`, cross-validation diagnostics, table-based sensitivity |
| [Regimes](api/regime.md) | Continuous-regime and bucketed configuration recommendations |
| [Visualization](api/viz.md) | `plot_front`, `plot_parallel`, `plot_scores`, `plot_calibration` |
| [I/O](api/io.md) | `load_results`, `save_results` |
| [Paired comparisons](api/paired.md) | `paired_difference`, `paired_rank` |
| [Decision reports](api/decision.md) | `preference_sweep`, regret, practical equivalence |
| [Proportion precision](api/proportions.md) | Wilson intervals and fixed-bank sizing |
| [Result artifacts](api/artifacts.md) | Versioned bundles and guarded provenance |

`run_sequential` and `ReplicationPolicy` supply fixed/adaptive replication with
recorded bounds, counts and uncertainty assumptions. `EvaluationCache` and grid
checkpoints support revision-aware reuse and resume.

See [capabilities and executable examples](guide/capabilities.md) for current interfaces,
optional dependencies, limitations and the checks that run in CI.
