# Adaptive sessions

Ask for configurations with `AdaptiveSession.ask(n)`, evaluate them in your own
processes, and return results with `tell(trial_id, scores)`. Use a journal `path`
for asking and telling across processes. Reopening retains the trial population
but restarts the sampler random stream; batching and reopening do not promise
the identical proposal sequence of an uninterrupted sequential run.

Report per-replicate values to retain Monte Carlo standard errors and counts.
Confidence constraints require at least two finite replicates, including for
non-objective constraint scores. A rejected tell leaves its trial pending so it
can be corrected. Result score columns retain weighted objective means; raw
means and uncertainty remain in metadata, and feasibility checks use raw units.

Inspect `session.trials()` or filter with `trials("pending")`,
`trials("complete")`, and `trials("failed")`. Snapshots include configuration,
trial id, state, and metadata. Record evaluation failures explicitly:

```python
session.fail(trial_id, "worker timed out")
retry_id, config = session.retry(trial_id, max_retries=2)
# Evaluate config again and report against retry_id, not trial_id.
session.tell(retry_id, scores)
```

Retries create new trial ids with the same parameters and retain the failed
attempt. The bound applies across a chain: to retry a failed retry, pass its id.
No retry happens automatically. Calling `retry` again on the same failed id
returns its existing child, even after reopening or completion. Serialize retry
requests for a given id; simultaneous writers may create duplicate children.
Duplicate `tell` or `fail` reports and transitions from terminal states are
rejected. Retrying an external evaluation may repeat its side effects; the
caller must make those operations safe to repeat.

Use `session.enqueue(config)` to evaluate a known complete configuration before
sampling new ones. Every call queues a distinct evaluation. Queued configurations
are visible through `trials("waiting")`; `ask()` supplies their evaluation ids.
Continuous bounds, log factors and categorical/discrete levels are validated.

To import completed observations, give both sessions the same explicit
`revision="simulator-v2-scorer-v1-data-v3-fidelity-high"` and matching factor,
objective (including weights/directions) and constraint definitions:

```python
new_session.warm_start(previous_session)
# Saved tables from previous_session.results() also retain the import schema:
new_session.warm_start(load_results("previous-results"))
```

Imports preserve raw means, standard errors, replicate counts and evaluation
provenance. They create completed trials without calling a simulator. Repeating
an import skips known evaluation identities, including after reopening and
through intermediate sessions; conflicting results for an existing identity
are refused. All rows are validated before any are imported. Serialize imports
into a destination session. A bare grid table lacks a verifiable session schema
and cannot be imported automatically. The revision is the caller's assertion of
matching simulator, scorer, data, randomness and fidelity, not evidence inferred
from the scores.

Journals now store a versioned schema and refuse incompatible definitions on
reopen. Nonempty legacy journals without that identity are also refused: use a
new journal or study name. Legacy observations require reconstruction and
validation against their original definitions; they are not silently adopted.

::: trade_study.AdaptiveSession

::: trade_study.SessionTrial
