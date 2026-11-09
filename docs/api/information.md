# Model Information

A discrimination study generates independent cohorts from each candidate
structure and lets a reader (a trained classifier, a Bayes-factor computation,
a stacking weight) give every structure a posterior probability for every
cohort. [`confusion_table`](confusion.md) reports how often a prespecified
verdict rule is right; `model_information` reports how much the reader's
posterior resolves the structure at all, as one number that can compare designs.

For any distribution `q(m | y)` that the reader reports,

    I(M; Y) >= H(M) + E[log q(M | Y)]

because the difference is `E_Y KL(p(M | Y) || q(M | Y)) >= 0`. The estimate is
the stratified mean `H(pi) + sum_m pi_m * mean log q_i(m)` over the replicates of
each structure `m`, in nats (`bound_bits` divides by `log 2`), with a
normal-approximation interval and the score of each structure.

```python
info = model_information(posterior, truth, labels=["A", "B"],
                         failed="prior")
info.bound, info.low, info.high   # nats; at most info.prior_entropy
info.fraction_resolved            # share of the prior entropy resolved
info.by_truth                     # which structure the reader recognises worst
```

Conditions that the caller establishes:

- **Held out.** The posteriors are for replicates the reader was not fitted on.
  On training replicates the bound is optimistic.
- **Independent replicates** drawn from the structures with the frequencies
  `prior` (uniform by default); the number of replicates per structure may
  differ from `prior`, and at least two are needed for every structure with
  positive prior.
- **Calibration** is not required for validity: for any reader, calibrated or
  not, the bound does not exceed `I(M; Y)`; miscalibration lowers it. A
  calibrated reader is what makes the bound close to `I(M; Y)`.
- **Failures** are never dropped: an all-NaN row raises unless
  `failed="prior"`, which replaces it by the prior (zero information).
- **Failed strata.** If every replicate of a structure fails and
  `failed="prior"` is used, that structure contributes no information and a
  standard error of zero; check `n_failed` beside the interval.
- **Floor.** A probability for the true structure below `floor` is raised to
  it; `n_floored` counts them and any is a warning, because a probability of
  zero would make the bound `-inf`.
- The bound describes the simulated generating structures. It says nothing
  about the real data when none of the structures generated them.

## Along a design size

A design study needs the bound as a function of a size (units per cohort,
sampling days, follow-up) and a rule that picks the smallest adequate size, as
`power_curve` and `minimum_adequate` do for success indicators.
`information_curve` reads raw replicate rows in which the generating structure
is a configuration factor and the reader's log probability of it on a held-out
replicate, `log q(truth | y)`, is an observable; `minimum_adequate_information`
selects the smallest size that reaches a target in nats (or as a fraction of
`H(M)`).

```python
curves = information_curve(rows, "log_score", truth="model", along="n_units",
                           labels=["A", "B"], adjustment="bonferroni")
minimum_adequate_information(curves, 0.5, fraction=True)
```

The conditions are those of `model_information`, per size: held-out
replicates, independent, drawn with the declared frequencies, at least two per
structure at every size. Scores must be finite and at most 0: a failed
replicate is scored by the caller (for example with the log of the prior
probability of its structure, zero information), never dropped. The probability
floor is the caller's responsibility because the score is computed outside the
library. As for power curves, the lower end of a marginal interval holds at the
selected size only; `adjustment="bonferroni"` makes the statements across the
scanned sizes simultaneous (approximately), and `monotone_above` reports a
curve that dips below the target after the selection. A selected size is a
statement about the simulated structures and the scanned grid; confirm it with
fresh replicates at the selected size.

::: trade_study.ModelInformation

::: trade_study.TruthScore

::: trade_study.model_information

::: trade_study.InformationPoint

::: trade_study.InformationCurve

::: trade_study.AdequateInformation

::: trade_study.information_curve

::: trade_study.minimum_adequate_information
