# trade-study

Multi-objective trade-study orchestration: scoring, Pareto optimization,
and Bayesian stacking.

For installation and quick-start examples, see the
[README](https://github.com/jcm-sci/trade-study#readme).

## Overview

`trade-study` provides a structured workflow for multi-objective
design-of-experiments studies:

1. **Define** observables and design factors via lightweight
   [Protocols](api/protocols.md)
2. **Build** experimental grids with [Design](api/design.md) utilities
   (full factorial, Latin hypercube, Morris screening)
3. **Run** simulations across the grid with [Runner](api/runner.md)
4. **Score** posterior predictive accuracy with proper scoring rules
   ([Scoring](api/scoring.md))
5. **Filter** the Pareto front ([Pareto](api/pareto.md))
6. **Stack** models via Bayesian stacking ([Stacking](api/stacking.md))
7. **Orchestrate** multi-phase studies with [Study](api/study.md)

## API Reference

| Module | Description |
|--------|-------------|
| [Protocols](api/protocols.md) | Core types: `Observable`, `Direction`, `Scorer`, `Simulator`, etc. |
| [Design](api/design.md) | `Factor`, `build_grid`, `screen`, `reduce_factors` |
| [Runner](api/runner.md) | `run_grid`, `run_adaptive` |
| [Study](api/study.md) | `Phase`, `Study`, `top_k_pareto_filter`, `weighted_sum_filter`, `feasibility_filter` |
| [Scoring](api/scoring.md) | `score`, `coverage_curve` |
| [Pareto](api/pareto.md) | `extract_front`, `pareto_rank`, `hypervolume`, `igd_plus` |
| [Stacking](api/stacking.md) | `stack_bayesian`, `stack_scores`, `ensemble_predict` |
| [Visualization](api/viz.md) | `plot_front`, `plot_parallel`, `plot_scores`, `plot_calibration` |
| [I/O](api/io.md) | `load_results`, `save_results` |
