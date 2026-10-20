# Fixed-sample proportion intervals

Recovery, power and coverage estimates from repeated simulations have Monte
Carlo uncertainty. Record the success count and trial count alongside the
estimated probability. A mean of 0.8 from 30 independent trials carries much
less precision than a mean of 0.8 from 2,000 trials.

```python
from trade_study import proportion_interval, replicates_for_half_width

interval = proportion_interval(24, 30)
assert interval.estimate == 0.8
print(interval.low, interval.high, interval.k, interval.n, interval.method)

# Fix this count before observing outcomes.
n = replicates_for_half_width(0.05, confidence=0.95)
assert n == 381
```

`proportion_interval` defaults to the two-sided Wilson score interval without
continuity correction. It uses the closed-form solution given in the
[NIST handbook](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm),
requires no optional dependency, and has approximate frequentist coverage.
All-failure and all-success samples retain nonzero uncertainty: zero observed
failures does not establish a zero failure probability. The `estimate` is
always `k/n`; the Wilson interval is generally not centered on that estimate.

For the conservative Clopper–Pearson interval, install `trade-study[stats]`
(SciPy >=1.11) and select the method explicitly:

```python
exact = proportion_interval(24, 30, method="clopper-pearson")
```

Clopper–Pearson uses the inverse regularized beta function for its lower tail
and the [complemented inverse](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.betainccinv.html)
for its upper tail. Keeping the small tail directly avoids rounding a
near-one quantile argument to one. The method corresponds to
[`scipy.stats.binomtest(...).proportion_ci(method="exact")`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats._result_classes.BinomTestResult.proportion_ci.html).
Its conservative coverage means at least the nominal level under the stated
sampling assumptions, not an exact-width interval or a guarantee for a
particular simulation/model implementation.

Counts must be integers, including NumPy integer scalars, with `0 <= k <= n`
and `1 <= n <= 2**53`. Booleans, fractional counts, zero trials and invalid
confidence levels are rejected. The count limit states the float64 numerical
range; it is not a statistical recommendation to run that many trials.

## Planning precision

`replicates_for_half_width(h)` plans for Wilson half-width `(high-low)/2`
at the worst observed proportion, 1/2. With the two-sided normal quantile
`z`, the worst continuous half-width is `z/(2*sqrt(n+z**2))`. The helper
inverts this expression and rounds to a positive integer, correcting
floating-point rounding at the boundary. The continuous bound is conservative
for odd counts, where exactly half successes is unattainable.

This is the width about the interval's center, **not** the maximum distance
from `k/n` to either endpoint. The helper plans precision, not hypothesis-test
power, probability of passing a gate, simultaneous design-comparison coverage,
or Clopper–Pearson width. It does not determine when to stop an experiment
after looking at its results.

## Indicator summaries and DataFrame columns

For raw replicated grid results, request interval metadata explicitly:

```python
import numpy as np
from trade_study import ResultsTable

outcomes = np.array([1.0] * 24 + [0.0] * 6)
raw = ResultsTable(
    configs=[{"design": "a"} for _ in outcomes],
    scores=outcomes[:, None],
    observable_names=["recovered"],
    metadata=[{"design_point": 0, "rep": rep} for rep in range(len(outcomes))],
)
summary = raw.aggregate_replicates(
    proportion_observables=["recovered"],
    confidence=0.95,
    proportion_method="wilson",
)
stored = summary.metadata[0]["proportion_intervals"]["recovered"]
assert (stored["k"], stored["n"]) == (24, 30)
assert stored["scope"] == "fixed-sample-marginal"
```

The default `aggregate_replicates()` behavior and score/annotation columns
are preserved. Only explicitly named observables get intervals; a fraction
such as 0.8 reported within one replicate is not treated as a Bernoulli trial.
Selected raw scores must be finite 0/1 outcomes, with unique non-negative
integer `rep` identities within each design. Fractional/non-finite scores,
missing identities, duplicate replicates and inconsistent parallel row counts
raise errors. No rows are dropped to reduce the denominator.

Install `trade-study[dataframe]` and call `summary.to_dataframe()` to obtain
columns such as `meta.proportion_intervals.recovered.k`, `.n`, `.estimate`,
`.low`, `.high`, `.confidence`, `.method` and `.scope`. These records can also
be stored in a versioned result bundle along with the caller's provenance.

## Sampling assumptions and interpretation

These intervals apply to independent, identically distributed Bernoulli
trials with a prespecified sample size, for one probability at a time.
The caller must establish those assumptions: a table of counts or distinct
replicate identifiers cannot establish independence. In particular:

- Dependent/clustered replicates, weighted samples and importance-sampling
  estimators require their own uncertainty methods.
- Common random numbers can preserve within-design trial independence while
  inducing cross-design dependence. These marginal intervals do not provide
  the uncertainty of a paired design difference; use an appropriate paired
  analysis for that question.
- Prespecify treatment of failures and the denominator. Non-finite recorded
  outcomes are rejected; entirely absent trial rows cannot be inferred from
  the table. Record planned versus actual trials and failure handling in
  study provenance rather than silently selecting successful runs.
- Repeated checking, result-dependent stopping, selecting the best design
  and simultaneous comparisons do not inherit the nominal fixed-sample
  coverage. Use the separately documented [sequential runner](../api/runner.md)
  for its supported interval/stopping contract. These fixed-sample helpers do not change
  `run_sequential`, `Constraint.confidence` or adaptive selection behavior.

Monte Carlo uncertainty describes the replication experiment. It does not
certify model adequacy, numerical accuracy or real-world recovery probability.
