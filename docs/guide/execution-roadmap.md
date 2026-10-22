# Execution and statistical integration plan

Engineering decisions recorded on 2026-10-21. Keep NumPy as the core dependency
and optional integrations behind extras. The shared
[execution roadmap](https://github.com/ACCIDDA/op_engine/blob/main/docs/development/execution-roadmap.md)
and [baseline protocol](https://github.com/ACCIDDA/op_engine/blob/main/docs/development/execution-baselines.md)
describe the numerical packages and representative measurements.

## Preserve package responsibilities

trade-study owns generic replicate orchestration, scores, result tables,
comparisons, optional surrogates and execution services. A simulator may use
NumPy, JAX, Torch or CuPy internally while returning explicitly declared host
results. This does not require making every runner/table operation GPU-native.
Inference-specific priors, measurement models, sensitivities and posterior
updating belong in OP Infer or a caller-supplied adapter.

No Rust rewrite or full Julia port is committed by this plan. Existing Julia
repositories are inactive design scaffolds. Native array transfer and host
serialization boundaries must remain explicit. Generic runners should retain
failures and stable task identities regardless of a simulator's backend.

## Fisher feasibility screening

The [bounded Fisher adapter](fisher.md) supplies a scalar result to existing
constraint machinery and preserves unavailable outcomes. It consumes an explicit
information artifact or caller-owned
screen instead of adding a compiler/inference dependency to the core. The
artifact records parameter coordinates/scales, observation noise, design,
nuisance treatment, reference parameter regime and numerical tolerances.

Use measured-output parameter sensitivities, not the instantaneous RHS state
Jacobian. Structural gauge freedoms need declared treatment; a known singular
nominal Fisher matrix is not grounds for rejecting every otherwise useful design.
Local numerical rank is a screen, not a structural-identifiability proof. Estimated
screen uncertainty needs an actual sampling/replication contract before a
probabilistic feasibility claim.

The implemented fixed-replicate allocation and finite-horizon bounded-score
methods remain available. Recoverable failed-attempt artifacts and gate-specific
power/false-failure qualification retain their separate scope. The Fisher guide
also distinguishes SBC, effect recovery, predictive checks and simulation-support
evidence; each needs its own declared reference and denominator.

## Predictive stacking and dimension penalties

The existing Bayesian path delegates to ArviZ predictive stacking. Preserve that
objective. Subtracting `IDOF * log(n)` from LOO scores is not an established
correction and must not become an automatic behavior of `stack_bayesian`.
Nominal parameter count, a reported gauge-quotient dimension, identifiable
dimension and singular-model information penalties are distinct quantities.

A proposed penalized alternative first needs a named objective, normalization,
unit of comparison, definition of sample size, assumptions and independent
qualification. Identical predictive distributions should give the same ensemble
prediction under duplicate representations; their individual weights can be
nonunique. Deduplication/grouping is a separate policy from complexity penalties.

See [predictive stacking](https://arxiv.org/abs/1704.02030) and
[singular-model information criteria](https://jmlr.org/papers/v14/watanabe13a.html).

## Surrogate acquisition and EVSI

Expected improvement over a current best score is a Bayesian-optimization
acquisition target. Decision-theoretic expected value of sample information
requires actions, utilities, a distribution of future observations and posterior
updating before optimizing the action. A Gaussian surrogate alone does not make
general EVSI analytic.

Keep those APIs and evidence separate. A generic acquisition may consume a
surrogate, while OP Infer or the caller supplies statistical updating and utility
semantics for EVSI. The current RF tree spread is a relative disagreement measure,
not calibrated posterior uncertainty. Qualification needs analytic decision
fixtures, retained simulation failures and Monte Carlo error before claims about
decision value. See the [EVSI formulation](https://doi.org/10.1137/19M1284981).

## Sequence

The high-priority [conditional table-sensitivity](table-sensitivity.md) increment
is implemented: pooled varying omitted inputs are refused, observed-stratum
conditioning and retained scope/validation are available, and direct discrete
contrasts reuse existing paired helpers. Configuration-factor grouping and
explicit dataset/replicate partitions answer distinct validation questions.
The [Sobol budget](../api/design.md#sobol-evaluation-budgets) records actual callback
costs before consumers price another screen. Remaining statistical and solver
extensions keep their separately qualified scope.

First reconcile overlapping issues and establish numerical-package baselines.
Then integrate forward/provenance and shared statistical interfaces. Add Fisher
screen adapters after sensitivity contracts are qualified. Keep penalized
stacking and EVSI implementation gated on their separately specified targets.
Forecasting consumers reuse trade-study execution and scoring rather than
introducing a second replicate runner.
