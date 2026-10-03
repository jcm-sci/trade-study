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

::: trade_study.AdaptiveSession
