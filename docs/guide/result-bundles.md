# Versioned result bundles

`save_result_bundle` writes an opt-in version-1 scientific result bundle to a
**new destination**. `load_result_bundle` returns its `ResultsBundle`: a scored
`ResultsTable`, ordered `Observable` definitions, provenance and schema version.
`load_results` returns only the scored table from either this format or a
consistent legacy export.

```python
import numpy as np
from trade_study import (
    Direction, Observable, ResultsTable, load_result_bundle, save_result_bundle,
)

results = ResultsTable(
    configs=[{"design": "a", "replicate": 0}],
    scores=np.array([[0.25]]),
    observable_names=["error"],
    metadata=[{"stream_id": [0, 7, 2], "root_seed": 2026}],
)
save_result_bundle(
    results, "new-result-bundle",
    observables=[Observable("error", Direction.MINIMIZE)],
    provenance={
        "model_id": "synthetic-decay-v1",
        "scorer_id": "absolute-error-v1",
        "score_transforms": {"error": "identity"},
        "environment_id": "record-your-environment-revision",
    },
)
bundle = load_result_bundle("new-result-bundle")
assert bundle.schema_version == 1
assert bundle.observables[0].direction == Direction.MINIMIZE
```

Observable definitions must match score column names in order. Directions and
finite nonnegative weights are preserved independently of raw score values.
The table itself does not carry scorer/model identities, learned transforms,
external data revisions or stream-allocation policies: supply those through
the primitive JSON `provenance` mapping. Available per-row replicate/seed
metadata stays in `ResultsTable.metadata`. The bundle records trade-study and
NumPy versions in `provenance["runtime"]`, with caller information separately
in `provenance["caller"]`.

## Version-1 schema and validation

Required files are `scores.npz` and `meta.json`; `annotations.npz` is required
exactly when the manifest declares an annotation array. Each NPZ contains only
its named array (`scores` or `annotations`). The metadata has exactly:

| Field | Contract |
| --- | --- |
| `schema_version` | Integer `1`, with unknown versions rejected. |
| `encoding` | `primitive-json-v1`. |
| `configs` | One string-keyed mapping per score row. |
| `observable_names` | Unique nonempty score column names, in order. |
| `annotation_names` | Unique nonempty annotation column names, in order. |
| `metadata` | Empty, or one string-keyed mapping per score row. |
| `observables` | Ordered names, `minimize`/`maximize` directions and weights. |
| `provenance` | Runtime version strings and caller-managed primitive JSON identities. |
| `arrays` | Score and optional annotation file descriptions. |

An array description records its fixed filename, SHA-256 digest, two-dimensional
shape and NumPy dtype string. Readers hash the same file bytes they decode,
load without pickled objects, and check the decoded shape/dtype against the
description. Scores and annotations must be real numeric matrices matching
the row counts and column names. Non-finite score values can represent existing
failed/undefined scores and remain in the numeric arrays; provenance/config
JSON must be finite. Empty tables with correctly shaped matrices are supported.

Hashes detect inconsistent/replaced array payloads. They are not signatures or
independent verification of model identities, provenance or scientific claims.

Supported input values are Python `None`, strings, booleans, integers, finite
floats, lists and dictionaries with string keys, nested in any combination.
These values round-trip without type conversion. Tuples, Paths, NumPy scalar
objects/arrays, enums, callables, non-finite JSON numbers and other opaque
objects are rejected before destination creation. Convert them to an explicit
portable representation under your own input schema when needed.

Preserving inputs and scientific interpretation does not reconstruct executable
simulators/scorers or certify a full replay. Record source/environment versions,
external inputs, model/scorer definitions, transforms and random-stream
allocation alongside the bundle. This format is separate from the transactional
study checkpoint identity and recovery mechanism.

## Initial publication and legacy compatibility

The first writer validates inputs and snapshots arrays before reserving a new
directory. It writes array files, then publishes `meta.json` with one atomic
same-filesystem replacement. Before metadata is committed, a reader cannot
load the destination as a complete bundle. An interrupted save may leave an
incomplete directory, but a retry refuses to overwrite it. Existing directories,
including empty or incomplete ones, raise `FileExistsError` and retain their
contents. Callers must keep inputs stable during serialization and exclusively
own the new destination. Atomic rename visibility is assumed; power-failure
durability, concurrent mutation of bundle files and distributed/object-store
publication are outside this first increment.

Existing `save_results` keeps its legacy multi-file export and `default=str`
fallback for unsupported config/metadata objects. It can overwrite a directory
and is not an atomic update. Legacy bundles remain readable when rows, names,
array dimensions and metadata agree. They return schema version `0`, empty
observable definitions and empty provenance: missing interpretation is explicit.
Malformed legacy arrays/metadata now raise `ValueError` rather than producing
a mismatched result table. A missing version field in otherwise versioned
metadata is rejected, without silently downgrading to a legacy interpretation.

Legacy `save_results` refuses to overwrite a committed versioned bundle, so a
different export call cannot silently discard its interpretation and integrity
metadata. Save legacy exports to their own destinations.

To migrate, load a consistent legacy table, explicitly recover any lossy input
types, supply the scientific definitions/provenance and save to a new version-1
destination. Repeated-generation updates with an atomic current-generation
pointer are the next focused increment of
[issue #172](https://github.com/jcm-sci/trade-study/issues/172).
