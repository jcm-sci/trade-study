# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

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
