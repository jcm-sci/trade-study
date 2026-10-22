# Conditional table sensitivity and discrete effects

`sensitivity_from_table` estimates the sensitivity of a **fitted response
surface**. Sobol reports first-order variance indices; Morris reports elementary
effects. Neither method in this wrapper assigns an index to discrete capacity
`q`, categorical rules or profiles. A pooled analysis that drops those inputs can
hide their effects or attribute them to correlated continuous settings.

Varying discrete/categorical factors are refused by default. Provide all factor
definitions and either analyze a table where those factors are observed constants
or use `condition_on` to select an actually observed stratum. Remaining varying
discrete/categorical inputs are still refused. Every observed constant, including
a continuous factor made constant by conditioning, is bound in prediction and
excluded from screening. At least one continuous factor must vary in that stratum.
Missing factor values and unobserved/invalid conditions fail explicitly.

Varying undeclared configuration keys are also refused: leaving `q` out of the
factor list must not quietly hide it. Genuine config bookkeeping such as a seed
can be explicitly listed in `metadata_keys`; these exclusions are reported in
`omitted_config_keys`. This declaration asserts that the key is not an input
whose effect is being studied. It cannot override a declared model factor. Put
trial bookkeeping in `ResultsTable.metadata` where practical. Undeclared constant
config fields are reported as omitted, without inventing a factor definition.

## Complete synthetic workflow

This factorial has a known `q × log(prior)` interaction plus a continuous `z`
effect. The replicate noise is deliberately shared across designs, so paired
differences cancel it exactly. This is an analytic fixture, not uncertainty or
coverage qualification for a stochastic simulator.

Install the optional `trade-study[design,surrogate]` dependencies.

```python
import math
import numpy as np
from trade_study import Factor, FactorType, ResultsTable, sensitivity_from_table
from trade_study.paired import paired_difference

configs, scores, metadata = [], [], []
design = 0
for q in (2, 4):
    for prior in (1.0, 10.0, 100.0):
        for z in (0.0, 0.5, 1.0):
            for rep in range(4):
                configs.append({"q": q, "prior": prior, "z": z})
                scores.append([(q - 2) * math.log10(prior) / 2 + z + 0.01 * rep])
                metadata.append({"design_point": design, "rep": rep})
            design += 1
table = ResultsTable(configs, np.asarray(scores), ["loss"], metadata=metadata)
factors = [
    Factor("q", FactorType.DISCRETE, levels=[2, 4]),
    Factor("prior", FactorType.CONTINUOUS, bounds=(1.0, 100.0), log_scale=True),
    Factor("z", FactorType.CONTINUOUS, bounds=(0.0, 1.0)),
]
conditional = sensitivity_from_table(
    table, factors, condition_on={"q": 4}, cv_group_by="design", cv_folds=3,
    n_estimators=30, n_trajectories=32, seed=7, warn_below_r2=None,
)
assert conditional.screened_factors == ("prior", "z")
assert conditional.fixed_factors == {"q": 4}
assert conditional.source_rows == tuple(range(36, 72))
assert conditional.validation_scheme == "held-out-designs"

effects = {}
for prior in (1.0, 10.0, 100.0):
    comparison = paired_difference(
        table, {"q": 4, "prior": prior, "z": 0.5},
        {"q": 2, "prior": prior, "z": 0.5}, "loss", method="t",
    )
    effects[prior] = comparison.mean
assert np.allclose(list(effects.values()), [0.0, 1.0, 2.0])
```

The direct differences vary with prior setting, exposing the interaction.
Each comparison fixes **every other design factor** and pairs genuine shared
replicate identities. `paired_rank` can compare several capacities within such
a fixed setting after selecting that subtable. Ranking an entire `q × prior`
grid instead compares joint designs. Shared integer labels alone do not establish
common-random-number or independent-replicate assumptions in a real application.
Retain failures, unmatched/nonfinite pairs, the actual paired denominator and
the chosen interval/multiplicity policy. The helper's finite-pair handling does
not justify treating failed runs as missing at random.

Conditional Sobol indices at different `q` values are separately normalized by
different conditional variances. Their difference is **not** an estimate of the
effect of `q` or of its interaction. Use direct factorial/paired contrasts for
those questions. Do not relabel `q` as continuous to bypass the refusal.

## Validation and target distribution

`cv_group_by="design"` holds out complete identical declared configurations,
keeping their repeated rows together. `cv_group_by=["regime_factor", ...]`
holds out groups defined by declared configuration factors; this can test unseen
regimes when those factors and the selected stratum support that interpretation.
Both group options unset retain shuffled row validation. These answer different questions, and
row validation can look strong when related designs/regimes generalize poorly.

Design-based grouping keeps repetitions of one configuration together. A shared
simulated dataset reused across different configurations can still cross its
folds. To hold out complete shared datasets/replicates, supply an explicit
`cv_groups` vector aligned with **every original table row**, before conditioning.
The wrapper validates the complete vector, then selects its `source_rows`.

```python
shared = sensitivity_from_table(
    table, factors, condition_on={"q": 4},
    cv_groups=[meta["rep"] for meta in table.metadata], cv_folds=2,
    n_estimators=20, n_trajectories=32, seed=7, warn_below_r2=None,
)
assert shared.cv_group_source == "external"
assert shared.validation_scheme == "held-out-external-groups"
assert shared.validation_groups == [0, 1, 2, 3] * 9
```

`cv_groups` and `cv_group_by` are mutually exclusive. Explicit labels can come
from a dataset ID, replicate ID or another caller-declared partition; no meaning
is inferred from `metadata_keys` or automatic table-field discovery. Labels may
be finite numbers, strings, bytes or tuples of those. Missing/nonfinite labels,
unsupported objects, unordered/mapping vectors and length/shape mismatches are rejected, including in rows
excluded by conditioning. Equal numeric labels such as `1`, `1.0` and NumPy scalar
equivalents share a group, also inside tuples.

The result retains external labels, stable integer group IDs and
`cv_group_source`. Holding out datasets tests a different target from holding out
designs. This route supplies **one partition**: grouping `(design, replicate)`
pairs does not independently keep all designs and all replicates together.
Crossed dependencies or other split policies need a separately justified
validation design. Grouping does not validate its own scientific assumptions.

`TableSensitivity` preserves selected original `source_rows`, screened and fixed
factors, requested conditions, excluded config keys, actual grouping names/ids,
grouped `surrogate_cv_r2`/`surrogate_cv_rmse`, external source/labels where selected,
and shuffled-row companions. With
fewer than two surviving groups for an observable, grouped metrics remain NaN;
row metrics never replace them. `cv_folds_requested` is a maximum, not a promise
that every observable has that many usable folds. Nonfinite scores retain the
existing per-observable surrogate filtering, so available groups can differ.

The recorded `sampling_factors` define independent uniform sampling over each
declared linear coordinate or over its log coordinate for `log_scale=True`.
Sobol/Morris queries can therefore use combinations never present in the saved
table. `support_ranges` retains marginal observed ranges per fitted observable,
and sampling beyond those ranges emits a warning once per observable. Marginal
range coverage and good CV do not establish **joint** response-surface support,
validity outside observed combinations, or an appropriate input distribution.
Restrict the declared bounds or collect a suitable design when needed. A broad
mixed-factor estimator, dependent-input sensitivity, noise decomposition and
uncertainty for the estimated indices require separate qualification.

Existing continuous-only calls retain their default method and row CV, while
returning explicit scope/validation information. Calls that previously silently
dropped varying inputs now fail; supply an observed stratum or use direct
contrasts. Existing frozen studies and environments need not be migrated.

Required CI also runs the scope, GP/RF and grouped-surrogate checks on a named
older CPU stack: Python 3.11, NumPy 1.24.0, scikit-learn 1.3.2, SALib 1.5.0,
SciPy 1.11.4 and pandas 2.0.3. Missing dependencies or skipped selected cases fail
that job. Current supported Python jobs resolve their own optional dependencies;
this compatibility point does not qualify every intervening version.
