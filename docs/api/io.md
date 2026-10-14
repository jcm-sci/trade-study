# I/O

Serialization and deserialization of study results.

`save_results()` supports replacing a result in an existing directory. It
replaces scores and metadata, writes the new annotation array when present,
and removes any previous `annotations.npz` when the replacement is unannotated.
Unrelated files remain. These writes are not atomic across the result files;
use separate output directories if readers need an unchanged previous bundle
during a save.

`load_results()` also reads [versioned result bundles](../guide/result-bundles.md),
returning their scored table. Both readers now reject inconsistent parallel
rows, names, annotation dimensions and metadata cardinality. Versioned files
also require matching digests/shapes/dtypes. For preserved directions and
provenance, use `load_result_bundle()`.
Legacy `save_results()` refuses a committed versioned destination.

::: trade_study.load_results

::: trade_study.EvaluationCache

::: trade_study.save_results
