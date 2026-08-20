# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added

- Replicated trials: `run_grid(..., n_reps=N)` evaluates each design point N times; simulators may opt in to per-replicate randomness via an optional `rep` keyword on `Simulator.generate` (detected by introspection). `ResultsTable.aggregate_replicates()` collapses replicate rows back to per-design-point means with `n_reps`/`score_std` metadata. `Phase.n_reps` forwards this into `Study`, and phase filtering now runs against aggregated design points rather than raw replicates when `n_reps>1` (#112).
- Surrogate accuracy reporting: `fit_surrogate()`/`SurrogateModel` (and `fit_regime_surrogate()`/`RegimeSurrogate` via passthrough) now compute a uniform k-fold cross-validated `cv_r2`/`cv_rmse` per observable for both the `gp` and `rf` backends. Warns (`warn_below_r2`, default threshold `0.0`) at fit time and in `RegimeSurrogate.recommend()` when an observable's accuracy is too low to trust (#114).
- `sensitivity_from_table()`: post-hoc Sobol/Morris sensitivity from an already-collected `ResultsTable`, by fitting a cheap surrogate over it (`fit_surrogate`) and running `screen()`'s existing machinery against the surrogate instead of a fresh simulator. Unlike a marginal Spearman correlation, this correctly detects non-monotonic (e.g. U-shaped) factor effects. Returns a `TableSensitivity` with `importance` indices and the surrogate's `surrogate_cv_r2` (#114) so callers can judge whether to trust the result (#113).
- `sobol_indices()`: like `screen(method="sobol")`, but returns both S1 (first-order) and ST (total-order) per observable instead of discarding ST. `screen()` itself is unchanged for backward compatibility; `ST - S1` is the standard way to detect interaction effects that a first-order-only view misses entirely (#120).
- Replicate averaging in `run_adaptive()` and `screen()`/`sobol_indices()` (`n_reps`, #122): the same `run_grid(..., n_reps=N)` convention (#112), now applied consistently across every entry point that repeatedly evaluates a simulator/`run_fn`. `run_adaptive` detects an opt-in `rep` keyword on `Simulator.generate` and averages each trial's objective(s) over `n_reps` draws before Optuna sees them; `screen()`/`sobol_indices()` detect the same convention on the bare `run_fn` callable they take instead. Without this, adaptive optimization or sensitivity screening against a stochastic simulator could select a "best" config or report an "important" factor that's actually just a single lucky/unlucky data draw. Default `n_reps=1` preserves prior behavior.

## [0.2.0] — 2026-08-17

### Added

- Regime-conditional surrogate (`RegimeSurrogate`, `fit_regime_surrogate`): interpolates recommended configs across continuous regime descriptors instead of hard-coded buckets (#109).
- Surrogate modeling (`SurrogateModel`, `fit_surrogate`): GP/RF interpolation over a results table for cheap approximate scoring (#108).
- `run_successive_halving` and `run_hyperband` runners for budget-constrained multi-fidelity search (#107).
- Coupled `FactorConstraint` support in `build_grid` for constrained QMC/LHS sampling (#106).
- Multi-fidelity studies via per-`Phase` `world`/`scorer` overrides, enabling cheap-then-expensive study designs (#101).
- `Constraint` dataclass and `feasibility_filter` for constraint-aware phase filtering (#99).
- Sobol sensitivity analysis in `screen()`, alongside the existing Morris method (#98).
- Progress callback support for `run_grid` and `Study.run` (#97).
- `Observable.weight` and `weighted_sum_filter` for weighted multi-objective aggregation (#96).
- Callable `grid` support in `Phase` for dynamic, state-dependent refinement (#95).
- Visualization module: `plot_front`, `plot_parallel`, `plot_calibration`, `plot_scores`, plus domain-specific example plots (#92, #93).
- New examples and guides: Bayesian model-criticism study (#100), monitoring-station design study.
- Documentation: complete `study`/`viz` API reference, PyPI/DOI badges (#102).

### Fixed

- `screen()` usage in the CSTR example now uses continuous factors, matching Morris/Sobol screening requirements.
- Docs: corrected KaTeX delimiters and snippet-directive rendering.

## [0.1.0] — 2026-04-15

### Added

- Core protocols: `Observable`, `Direction`, `Simulator`, `Scorer`, `TrialResult`, `Annotation`, `ResultsTable`.
- Design module: `Factor`, `FactorType`, `build_grid` (full factorial, LHS, Sobol, Halton), `screen` (Morris), `reduce_factors`.
- Runner: `run_grid` (joblib parallel), `run_adaptive` (optuna NSGA-II).
- Multi-phase orchestration: `Phase`, `Study`, `top_k_pareto_filter`.
- Scoring: `score` (CRPS, WIS, interval, energy, RMSE, MAE, coverage, Brier), `coverage_curve`.
- Pareto analysis: `extract_front`, `pareto_rank`, `hypervolume`, `igd_plus`.
- Bayesian stacking: `stack_bayesian`, `stack_scores`, `ensemble_predict`.
- I/O: `save_results`, `load_results` (NumPy `.npz` + JSON metadata).
- Input validation for `Factor` (empty name/levels, invalid bounds).
- PEP 561 `py.typed` marker for downstream type checking.
- User guides: CSTR reactor design, scikit-learn hyperparameter sweep.
- Examples: `cstr_study.py`, `sklearn_study.py`.
- CI: lint + type-check + test + examples workflows.
- Documentation: mkdocs-material site with API reference.
