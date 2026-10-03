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

::: trade_study.run_grid

::: trade_study.run_adaptive

::: trade_study.run_successive_halving

::: trade_study.run_hyperband
