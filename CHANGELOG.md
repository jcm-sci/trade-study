# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added

- Adaptive trial inspection, explicit failure reporting, and bounded retries with persistent failure reasons and retry lineage (#151).
- Incremental grid checkpoints preserve completed design-point/replicate evaluations across interruptions, including parallel workers and incomplete `Study` phases. `run_grid(max_retries=...)` provides opt-in bounded retries (#151).
- Grouped surrogate validation holds out whole designs or regimes alongside separate row-validation metrics. Prediction and recommendation expose observed-support diagnostics and warn on extrapolation for GP and RF (#152).
- `run_sequential()` allocates bounded extra replication to unresolved comparisons and feasibility boundaries, preserves raw replicate ids, and reports budgets, stopping reasons and simultaneous finite-horizon mean intervals under explicit bounded-score assumptions (#153).
- Adaptive sessions queue known configurations and import compatible completed observations with persistent identity/provenance and duplicate protection. Versioned session schemas validate reopened journals and refuse unverifiable legacy storage (#154).
- Opt-in `EvaluationCache` reuses grid evaluations by typed configuration, replicate namespace/id, model/scorer revision, fidelity, objective and annotation definitions; includes provenance, bypass, invalidation and conflicting-evidence checks (#154).
- `preference_sweep()` reports ranking/selection stability, regret, feasible Pareto alternatives, raw-unit practical equivalence and optional paired uncertainty under explicit normalization and preference assumptions, with exportable per-design summaries (#155).

### Fixed

- Queued, retried and imported adaptive evaluations now join NSGA-II generations and process constraints when completed; interrupted imports resume idempotently. The adaptive extra now requires Optuna >=4.5 for its public generation API (#164).

## [0.3.0] — 2026-10-02

### Added

- `ResultsTable.to_dataframe()` exports factor values, scores, annotations and optional flattened trial metadata; pandas remains optional via the `dataframe` extra (#85).

- Replicated trials: `run_grid(..., n_reps=N)` evaluates each design point N times; simulators may opt in to per-replicate randomness via an optional `rep` keyword on `Simulator.generate` (detected by introspection). `ResultsTable.aggregate_replicates()` collapses replicate rows back to per-design-point means with `n_reps`/`score_std` metadata. `Phase.n_reps` forwards this into `Study`, and phase filtering now runs against aggregated design points rather than raw replicates when `n_reps>1` (#112).
- Surrogate accuracy reporting: `fit_surrogate()`/`SurrogateModel` (and `fit_regime_surrogate()`/`RegimeSurrogate` via passthrough) now compute a uniform k-fold cross-validated `cv_r2`/`cv_rmse` per observable for both the `gp` and `rf` backends. Warns (`warn_below_r2`, default threshold `0.0`) at fit time and in `RegimeSurrogate.recommend()` when an observable's accuracy is too low to trust (#114).
- `sensitivity_from_table()`: post-hoc Sobol/Morris sensitivity from an already-collected `ResultsTable`, by fitting a cheap surrogate over it (`fit_surrogate`) and running `screen()`'s existing machinery against the surrogate instead of a fresh simulator. Unlike a marginal Spearman correlation, this correctly detects non-monotonic (e.g. U-shaped) factor effects. Returns a `TableSensitivity` with `importance` indices and the surrogate's `surrogate_cv_r2` (#114) so callers can judge whether to trust the result (#113).
- `sobol_indices()`: like `screen(method="sobol")`, but returns both S1 (first-order) and ST (total-order) per observable instead of discarding ST. `screen()` itself is unchanged for backward compatibility; `ST - S1` is the standard way to detect interaction effects that a first-order-only view misses entirely (#120).
- Replicate averaging in `run_adaptive()` and `screen()`/`sobol_indices()` (`n_reps`, #122): the same `run_grid(..., n_reps=N)` convention (#112), now applied consistently across every entry point that repeatedly evaluates a simulator/`run_fn`. `run_adaptive` detects an opt-in `rep` keyword on `Simulator.generate` and averages each trial's objective(s) over `n_reps` draws before Optuna sees them; `screen()`/`sobol_indices()` detect the same convention on the bare `run_fn` callable they take instead. Without this, adaptive optimization or sensitivity screening against a stochastic simulator could select a "best" config or report an "important" factor that's actually just a single lucky/unlucky data draw. Default `n_reps=1` preserves prior behavior.
- `recommend_bucketed_config()`: the discrete counterpart to `RegimeSurrogate` (#123). For a handful of named regimes too sparse (and too far outside any existing training data) for a surrogate to extrapolate across sensibly, runs `run_adaptive` independently per regime, picks each regime's best-found config by a primary objective, groups regimes into named buckets, and aggregates each bucket's per-regime best configs (median for continuous/discrete factors, mode for categorical) into one recommended config per bucket.
- `recommend_bucketed_config()` split into `recommend_per_regime()` (the expensive adaptive search) and `aggregate_bucketed_config()` (cheap post-processing), with `recommend_bucketed_config()` now a thin wrapper composing them. Lets a caller experiment with different `bucket_fn` groupings against the same search results without re-running `run_adaptive` for every attempt -- found necessary in practice deriving VBPCApy buckets, where a first grouping choice performed poorly and needed re-grouping without repeating an hours-long search.
- `stack_proportional()`: weights models proportionally to direction-aware exponential utilities of their mean score, with an explicit score-unit temperature, instead of `stack_scores()`'s linear-program optimum (which puts *all* weight on the single best model for any nonzero gap, however small -- a real problem in practice when two models are near-tied and the "winner" flips between runs on noise well within measurement uncertainty). Falls back to a uniform split when every model scores identically.
- Log-scale continuous factors: `Factor(..., log_scale=True)` samples a positive-bounded factor uniformly in log space in `build_grid` (Sobol, Halton, LHS), `run_adaptive` (optuna `log=True`), `screen()`/`sobol_indices()` (sensitivity indices then refer to the log scale), and surrogate encoding, for parameters that span orders of magnitude. `unit_to_value()`/`value_to_unit()` expose the mapping (#131).
- `AdaptiveSession`: batched ask/tell adaptive search for evaluations that run elsewhere, such as cluster job arrays. `ask(n)` proposes configs with trial ids; `tell(trial_id, scores)` accepts one value or the per-replicate values of each observable, optimizes the (weighted) means, and keeps standard errors and replicate counts with each trial; `results()` returns a `ResultsTable` with that metadata and each trial's constraint values. With `path`, the session lives in an optuna journal file so asking and telling can happen in different processes, and `Constraint`s are passed to NSGA-II. `run_adaptive` is now an in-process loop over a session and reproduces its previous trials for a fixed seed; replicate scores that are NaN are now ignored in the mean rather than failing the trial (#132).
- Uncertainty-aware choices (#115): `SurrogateModel.spread_batch()` returns the predictive spread (GP standard deviation, or the spread across random-forest trees as a relative, uncalibrated measure); `RegimeSurrogate.recommend(..., risk=k)` ranks candidates by `prediction + k * spread` (or `- k * spread` when maximizing) instead of the raw prediction. `Constraint(..., confidence=0.95)` makes `ResultsTable.feasible()` test a one-sided bound, `mean + z * se` for `<=`/`<` and `mean - z * se` for `>=`/`>`, using each row's Monte Carlo standard error from `metadata["standard_error"]` or from `aggregate_replicates()`'s `score_std` and `n_reps`; `AdaptiveSession` applies the same bound when guiding the search.
- Study checkpointing (#75): `Study.run(checkpoint_dir=...)` saves each completed phase and, on a rerun, loads phases already on disk instead of running them, replaying filters on the loaded results. Results produced elsewhere, such as reduced from cluster shards, can stand in for a phase by being written into its directory with `save_results()`. `Study.save()` and `Study.load()` write and read the same checkpoint; a checkpoint written for a different phase list is refused.
- `Study.compare_phases()` (#81): one row per completed phase with its design-point count, front size, hypervolume (against a given reference point, or by default the worst front value of each observable over all phases, padded by 10% of its range) and best value per observable, plus IGD+ against the previous phase in both directions: `igd_plus_gain` (how far the previous front falls short of this one) and `igd_plus_loss` (how far this front falls short of the previous one). Replicated phases are compared over design-point means, and rows with non-finite scores are left out of the fronts.
- Paired comparisons under common random numbers (#133): `paired_difference(results, design_a, design_b, observable)` matches replicates by `metadata["rep"]` and returns the mean per-replicate difference `a - b` with a percentile-bootstrap or paired-t interval (`PairedDifference`). Designs are given as a config subset or a `design_point` index; mismatched or repeated replicates and rows without `rep` are rejected, and pairs with a non-finite value are dropped. `paired_rank(results, observable, reference)` compares every design with a reference, best first, with Bonferroni-adjusted intervals by default.

### Fixed

- Checkpoints now validate a versioned study definition rather than phase names alone, reject incompatible/legacy manifests and phase schemas, and atomically write the manifest. `Study.checkpoint_key` identifies caller-managed model/data revisions and opaque callable behavior (#143).

- Adaptive phases now honor `Phase.n_reps`, retain raw means/standard errors/replicate counts, and filter/compare already-averaged trials without re-aggregating them. Feasibility checks use raw means in observable units even when adaptive scores are weighted (#142).

- Confidence constraints now reject non-finite means and unknown/invalid standard errors instead of treating missing uncertainty as zero. Adaptive tells require constraint scores and validate them before updating storage; rejected tells can be corrected (#141).

- `stack_proportional()` now uses direction-aware exponential utilities with an explicit score-unit `temperature`, preserving shared weight for near-tied losses as well as rewards; validates input and handles signed/extreme scores (#140).

- Strict mypy failed on `run_adaptive`'s optuna `directions` argument with current optuna stubs; directions are now typed as `Literal["minimize", "maximize"]`.
- `reduce_factors()` no longer lets a NaN-valued observable (e.g. a Type-I rate that's legitimately undefined outside null regimes) silently corrupt every other observable's importance for the same factor. It aggregated via `np.maximum`, which propagates NaN (`np.maximum(4.05, nan) == nan`); one conditionally-undefined observable could erase a real, significant importance value found via a *different* observable, dropping the factor with no warning or error. Now uses NaN-safe `np.fmax`, and warns when a factor's importance is NaN across *every* observable (dropped for lack of data, not confirmed unimportance) (#119).

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
