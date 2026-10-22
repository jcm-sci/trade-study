# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added

- Bounded local Fisher-screen adapter for caller-qualified host information
  matrices, explicit coordinates/target directions, existing feasibility
  constraints, distinct unavailable/failed outcomes and planned-ID counts.
  Export primitive JSON context/policy/evidence without new dependencies.

### Documentation

- Separate computational SBC, effect recovery, predictive adequacy and simulation
  support. Document PPC reference-law, conditional-denominator and sampling limits.

- Record execution ownership, host-result boundaries, Fisher-screen integration
  requirements and separate qualification targets for penalized stacking and EVSI.

## [0.6.0] - 2026-10-20

### Added

- Experimental frozen Slurm task plans (#174): detached commit checkouts,
  checked project import origins/runtime, immutable Python task argv and external
  data revisions, single-node CPU packing, per-task logs, locked atomic completion
  receipts and declared output hashes. Reuse native grid ledgers/adaptive journals
  inside tasks; refuse changed provenance/artifacts and rerun missing outputs.
  Check completed dependency exit codes before omitting afterok dependencies.
  Include a complete analytic normal-mean example and real Git/process/Bash
  qualification through a scheduler command fixture. A two-CPU Rockfish pilot
  also verified task execution, completed-job receipt reuse and failed-dependency
  refusal. Broader target-site qualification and direct distributed runner/session
  integration remain open.

### Documentation

- Replace issue and pull-request breadcrumbs in guides and API docstrings with
  current behavior, named interfaces and explicit limitations.
- Edit introductions and worked guides for plain language. Remove unsupported
  performance and optimizer comparisons, scope illustrative results to their
  evaluated designs, and state the Pareto dominance definition explicitly.

## [0.5.1] - 2026-10-19

### Fixed

- Solve `stack_scores`' linear supplied-score objective directly, selecting the
  best computed mean and splitting exact best-mean ties uniformly. Preserve
  the optimum under positive rescaling, reject invalid/empty/non-finite score
  matrices and avoid unscaled sum overflow. SciPy is no longer needed for this
  NumPy-core operation (#204).
- Normalize a separate weight array in `ensemble_predict`, accepting read-only
  arrays and views without mutating the caller or backing storage (#205).

### Documentation

- Add a generated precision-cost figure and summary table to the paired-study
  quickstart, with analytic MSE and Monte Carlo uncertainty clearly separated.
  Verify source, plotted data and export identities in example CI. Select Python
  3.12 explicitly when regenerating with the pinned optional plotting version.
- Audit current capabilities against public APIs; correct scope and limitations,
  add executable workflow examples and public API doctests, and document their
  CI and optional-dependency checks. Fix the Bayesian example energy-score axes
  and execute both monitoring and Bayesian guides in CI. Refresh affected tutorial
  figures and clarify positive Pareto scaling, score-weight optimization and
  empirical coverage interpretation. The documentation audit itself did not
  change core runtime APIs or dependencies; this patch release also includes
  the stacking fixes above.


## [0.5.0] - 2026-10-19

### Fixed

- `extract_front`, `pareto_rank`, `hypervolume` and `igd_plus` raise an
  `ImportError` that names `pip install 'trade-study[pareto]'` when pymoo is not
  installed, instead of a bare `ModuleNotFoundError` from inside the module
  (#200).

### Added

- `ReplicationPolicy(allocation="fixed")` collects its full prespecified count
  before one bounded-mean inference look; adaptive allocation remains the default.
  Sequential summaries export a versioned replication contract with used bounds,
  counts, objectives/constraints, multiplicity, stopping, precision and explicit
  unassessed gate power/false-failure rates. Used policies are detached from
  caller-owned mappings; cluster-score denominators and abort-on-failure behavior
  are explicit (#176, first policy/metadata increment; broader qualification and
  recoverable failure artifacts remain open).

- `examples/serosurvey_frames.py` adds a sampling-frame design factor to the
  serosurvey example: a community survey or residual specimens from facilities
  serving the same communities, with invented facility and specimen costs and a
  selection-odds model linking specimen provision to antibody status
  (`SurveyWorld(selection_odds=...)`, `sampling_frame`, `SurveyCosts`
  residual-frame costs). It runs the 36 designs at four selection-odds values,
  saves a result bundle with provenance, and reports the winners and the
  smallest subgroup error each frame reaches within the budget. With odds of
  one the residual frame reproduces the community survey exactly. The notebook
  and guide add a section citing the residual-specimen comparisons from India,
  and CI runs the script (#191).

## [0.4.0] — 2026-10-15

### Added

- The serosurvey example compares designs replicate by replicate: oversampling
  against proportional allocation at a size fixed in advance, with paired
  bootstrap intervals and the variance reduction from pairing
  (`allocation_contrast`, `plot_allocation`), and every feasible design against
  a reference design through `preference_sweep(paired_reference=...)`
  (`reference_rows`). The notebook and guide add a section on whether
  oversampling is worth it (#190).

- `examples/serosurvey_contexts.py` runs the serosurvey designs in nine
  contexts (intracluster correlation and underserved-group prevalence), keeps
  context and design keys separate, reports the winners, practically equivalent
  designs and steadiest design within each context and how often the winner
  changes along each context factor, and saves the per-design summaries as a
  version-1 result bundle with model, context, seed and objective provenance
  (`examples/data/serosurvey_contexts/`). The notebook loads the bundle, and CI
  runs the script (#188).

- The serosurvey example records a 0/1 indicator per simulated survey (the
  underserved-group estimate within ±5 percentage points of its target), sizes
  the refinement phase with `replicates_for_half_width` before running, and
  reports each feasible Pareto design's share with a fixed-sample 95% Wilson
  interval (`survey_indicators`, `planned_replicates`, `precision_rows`,
  `plot_precision`, `run_study`). The notebook and guide add a section on how
  often an estimate is close enough (#189).

- The serosurvey example declares raw-unit practical-equivalence tolerances and
  reports, for each feasible Pareto design, the priority scenarios it wins, its
  rank range, its largest regret and the scenarios for which it is practically
  equivalent to the winner (`acceptable_sets`, `robustness_rows`,
  `plot_robustness`); the priority figure outlines practically equivalent
  designs. The notebook and guide add a section on choices that hold up when
  priorities change (#187).

- Fixed-sample binomial proportion intervals record counts, estimates, bounds
  and methods: Wilson without optional dependencies and conservative
  Clopper–Pearson through the lazy `stats` extra. Worst-case Wilson replicate
  sizing plans interval half-width. Explicit indicator aggregation exports
  marginal interval metadata, rejecting fractional/non-finite outcomes and
  duplicate replicate identities; default aggregation is preserved. Sampling,
  denominator, multiplicity and sequential-stopping limits are documented (#175).

- Result publication retains immutable version-1 generations and selects the
  current generation with one atomic local-filesystem pointer replacement.
  Readers capture and validate a single generation, exposing its identifier;
  interrupted updates preserve the previous complete data. Direct/legacy files
  and older generations remain available, with explicit concurrency, retention
  and durability boundaries (#172).

- Opt-in version-1 result bundles preserve primitive JSON configs/metadata,
  ordered observable directions/weights and caller/runtime provenance. Readers
  validate parallel rows, names, numeric array shapes/dtypes and per-file
  digests, including legacy consistency checks. Initial bundle writes reserve
  new destinations and publish metadata last; repeated-generation publication
  remains the next increment of #172. Legacy stringifying export stays available.

- An executed IVAC-oriented Jupyter notebook compares synthetic serosurvey designs across cost, overall accuracy and underserved-group accuracy, with live budget/preference changes, a 15-minute presenter route and documentation figures. Install notebook tools with the `notebook` extra.
- Adaptive trial inspection, explicit failure reporting, and bounded retries with persistent failure reasons and retry lineage (#151).
- Incremental grid checkpoints preserve completed design-point/replicate evaluations across interruptions, including parallel workers and incomplete `Study` phases. `run_grid(max_retries=...)` provides opt-in bounded retries (#151).
- Grouped surrogate validation holds out whole designs or regimes alongside separate row-validation metrics. Prediction and recommendation expose observed-support diagnostics and warn on extrapolation for GP and RF (#152).
- `run_sequential()` allocates bounded extra replication to unresolved comparisons and feasibility boundaries, preserves raw replicate ids, and reports budgets, stopping reasons and simultaneous finite-horizon mean intervals under explicit bounded-score assumptions (#153).
- Adaptive sessions queue known configurations and import compatible completed observations with persistent identity/provenance and duplicate protection. Versioned session schemas validate reopened journals and refuse unverifiable legacy storage (#154).
- Opt-in `EvaluationCache` reuses grid evaluations by typed configuration, replicate namespace/id, model/scorer revision, fidelity, objective and annotation definitions; includes provenance, bypass, invalidation and conflicting-evidence checks (#154).
- `preference_sweep()` reports ranking/selection stability, regret, feasible Pareto alternatives, raw-unit practical equivalence and optional paired uncertainty under explicit normalization and preference assumptions, with exportable per-design summaries (#155).
- A synthetic assay-cost example and guide demonstrate categorical choices, external cost annotations, cost as a Pareto objective, and screening/refinement with replicated design means (#83).

### Changed

- The serosurvey simulator draws one finite population of communities per
  replicate from the seed and replicate id, and every design surveys it with the
  same participant draws (common random numbers), so smaller designs are nested
  in larger ones. The estimand is now the replicate's realized group prevalence
  (`SurveyWorld(estimand="fixed")` keeps the fixed group means). Figures, the
  notebook, the guide's numbers and the context bundle are regenerated (#190).

### Fixed

- Replicate aggregates store standard-deviation metadata as primitive Python
  floats, allowing finite summaries to round-trip through versioned result
  bundles without caller conversion. The artifact format continues to reject
  unsupported caller-supplied NumPy values (#184).

- Clean-wheel validation explicitly installs and tests with its temporary
  interpreter, preserving the editable development environment even when
  `UV_PYTHON` is set. A required CI job exercises that environment boundary (#180).

- Saving an unannotated result over an annotated result removes the stale
  annotation file, so reloaded scores/configurations cannot inherit old
  annotations. Overwriting annotated results still replaces their arrays and
  schema; unrelated files are retained (#171).

- Queued, retried and imported adaptive evaluations now join NSGA-II generations and process constraints when completed; interrupted imports resume idempotently. The adaptive extra now requires Optuna >=4.5 for its public generation API (#164).
- Strict adaptive `<`/`>` constraints now treat exact threshold equality as infeasible, matching result-table feasibility without rejecting adjacent feasible values (#167).

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
