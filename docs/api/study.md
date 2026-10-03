# Study

Multi-phase study orchestration.

::: trade_study.Phase

::: trade_study.Study

::: trade_study.top_k_pareto_filter

::: trade_study.weighted_sum_filter

::: trade_study.feasibility_filter

## Checkpoint identity

`Study.run(checkpoint_dir=...)`, `Study.save()` and `Study.load()` share a
versioned manifest. It checks phase order, grids, replication/trial counts,
factors, objectives, annotations, filters, and inspectable simulator/scorer
code. Incompatible definitions and legacy manifests are refused; use a new
checkpoint directory when intentionally changing the study.

Set `Study(..., checkpoint_key="model-v2-data-v1")` to identify simulator/scorer
instance configuration, external data, globals, and other behavior that cannot
be inferred from code. Update the key whenever those inputs change. Opaque
callables require a key. A matching key does not override structural mismatches.
A checkpoint is not a serialization of live simulator/scorer objects.

Grid phases also write a `trials.sqlite` ledger inside their phase directory.
An interrupted phase resumes its unfinished evaluations without repeating
completed design-point/replicate tasks. Adaptive phases retain phase-level
checkpointing; use a persistent `AdaptiveSession` for external adaptive trial
recovery. Final phase results keep the existing `save_results()` format.

Externally evaluated phase results can still be supplied with `save_results()`
after initializing the manifest using `Study.save()`. Their observable schema
must match the study.
