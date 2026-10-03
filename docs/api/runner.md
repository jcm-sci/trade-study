# Runner

Execute simulations across experimental grids.

For expensive grids, pass `checkpoint_path="run.sqlite"` and a caller-managed
`checkpoint_key="model-v1-data-v3"`. Successful evaluations are committed one
at a time, including by parallel workers. Resume with the same ordered grid,
replicate count, observable/annotation definitions, and revision. Completed
tasks are loaded; only unfinished tasks are evaluated. Configurations with
identical values at different grid positions remain separate design points.
Callbacks run for all rows, including recovered rows, in original task order.

`max_retries=2` permits two additional attempts for each unfinished task during
that invocation. The default is zero and propagates the first exception.
Simulator and scorer exceptions are retried with the same configuration and
replicate id. Interrupts are not retried. Metadata records total attempts across
invocations and marks recovered rows with `recovered=True`.

Checkpoint identity checks inspectable class code and structural definitions.
Change the revision when instance settings, external data, or randomness
semantics change. Use a new path for an incompatible definition. Run one grid
invocation at a time against a ledger; its workers may run in parallel. A crash
between an external side effect and saving its result may require repeating
that evaluation. Make side effects safe to repeat. Retry limits do not bound
the number of explicit resume invocations.

## Reusing identified evaluations

Reuse is opt-in through an explicit cache:

```python
cache = EvaluationCache(
    "evaluations.sqlite",
    revision="simulator-v2-scorer-v1-data-v3-annotations-v1",
    replicate_namespace="experiment-2026-seed-17",
    fidelity="high",
)
results = run_grid(world, scorer, grid, observables, n_reps=3, cache=cache)
```

The key includes a typed canonical configuration, replicate id, caller revision,
replicate namespace, fidelity, inspectable simulator/scorer class code,
objective definitions and annotation semantics. Grid position and total replicate
count are excluded: reordering designs or requesting more replicates can reuse
identical existing evaluations. Raw result rows still retain the current grid's
design-point ids. Different replicate ids and independent experiment namespaces
remain distinct even when their configurations match.

The caller must update the revision when instance settings, external data,
globals or opaque callable behavior changes. The replicate namespace must
identify the actual randomness/seed convention. Use a fresh namespace for an
independent experiment; matching configuration alone never establishes reuse.
Supported scalar/container configurations preserve types; opaque values are
refused. Cached evaluators must not mutate their configuration.

`cache_bypass=True` skips reads and writes, leaving stored evidence untouched.
`cache.clear()` invalidates all contexts in the cache file. Changing identity
creates a new context. Conflicting scores for one identity are refused.
Metadata exposes `cache_hit`, `cache_key`, and the revision/namespace/fidelity
context. Cached wall time is the original evaluation's time. Completed grid
checkpoints can populate the cache without re-evaluation; bypassing the cache
does not bypass checkpoints. Concurrent workers may evaluate an absent identity
more than once before either saves; reuse does not promise exactly-once effects.

::: trade_study.run_grid

::: trade_study.run_adaptive

## Sequential replication

`run_sequential(..., policy=ReplicationPolicy(...))` starts every design with
`min_reps`, then allocates additional replicates among unresolved designs,
preferring the least sampled design with grid-order tie breaking. It retires
confidently infeasible or dominated designs and stops when it resolves a single
choice, a Pareto trade-off, or practical equivalence. Otherwise it reports
`max_reps` or `budget_exhausted`; a budget stop does not establish a winner.

Specify known almost-sure `score_bounds` for every observable, `min_reps`,
`max_reps`, `max_evaluations`, and joint `confidence`. Optional `equivalence`
tolerances use raw observable units. Directions guide comparisons; observable
weights do not transform scores. Monotone constraints use the simultaneous
intervals and policy confidence instead of normal standard-error bounds.

For D designs, M observables, horizon H=`max_reps`, and alpha=`1-confidence`,
the interval radius after n replicates is
`(upper_bound - lower_bound) * sqrt(log(2*D*M*H/alpha)/(2*n))`, clipped to the
declared bounds. Hoeffding's inequality bounds each failure by alpha/(D*M*H);
a union bound gives simultaneous coverage over every design, observable, and
allowed sample size. This permits data-dependent allocation and stopping under
independent replicates with a fixed mean within each design. Common random
numbers may correlate designs; the within-design replicate sequences must
still satisfy those assumptions. Bounds must be known before inspecting data,
not estimated from the observed minimum and maximum. Out-of-bounds/non-finite
scores are errors. The bounds are conservative and may spend the entire budget
on small gaps, even for a deterministic simulator with broad score bounds.
For context on time-uniform inference and sharper approaches, see
[Kuchibhotla and Zheng (2021)](https://proceedings.mlr.press/v139/kuchibhotla21a.html).

`SequentialResult.results` preserves raw scores and design-point/replicate ids;
`summary` contains design means and exportable allocation/stopping metadata.
Use `intervals` and `feasibility` for the sequential decision. Ordinary
`ResultsTable.feasible()` normal bounds and post-selection paired bootstrap/t
intervals do not inherit sequential coverage. Allocations may have unmatched
tail replicates; explicitly select a common prefix for descriptive paired
comparisons. Existing `run_grid(n_reps=...)` behavior is unchanged.

::: trade_study.ReplicationPolicy

::: trade_study.SequentialResult

::: trade_study.run_sequential

::: trade_study.run_successive_halving

::: trade_study.run_hyperband
