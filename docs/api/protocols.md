# Protocols

Core types and interfaces for trade-study workflows.

::: trade_study.Direction

::: trade_study.Observable

::: trade_study.Annotation

::: trade_study.Scorer

::: trade_study.Simulator

::: trade_study.PartialEvaluator

::: trade_study.TrialResult

::: trade_study.ResultsTable

## DataFrame export

Install `trade-study[dataframe]` and call `results.to_dataframe()` to get one
row per trial with factor, observable, annotation, and metadata columns.
Metadata is flattened under `meta.` so replicate counts and uncertainty remain
available, including `meta.standard_error.<observable>` and raw adaptive means
in `meta.scores.<observable>`. Use `include_metadata=False` for only factors,
scores, and annotations. Conflicting column names raise an error instead of
silently overwriting values. Use pandas `frame.to_csv(path, index=False)` to
export CSV. Call `aggregate_replicates()` explicitly first if you want grid
results collapsed to design points.
