# Local information feasibility screens

`screen_fisher` consumes a caller-qualified information matrix and exposes a
local minimum-eigenvalue screen through the existing `Constraint` API. Core
dependencies remain NumPy-only. The inference layer or project supplies measured
output sensitivities, the observation/noise model, coordinate transformations and
any nuisance profiling. For example, OP Infer's bounded Gaussian information
adapter can supply its explicitly transferred host matrix; its qualification
scope and evidence references travel with the screen.

The optional `examples/fisher_op_stack.py` exercises the actual compiler,
forward sensitivity solver and Gaussian information adapter against an
independent exponential-decay derivative, including an initial-state parameter
and direct measurement parameter. A separate required NumPy CPU CI job installs
immutable reviewed public producer commits and runs that example. Those packages
remain optional for users of the core screen; no device qualification is implied.

For independent unit-variance measurements of `a + b*x`, a balanced design at
`x=-1,+1` has information `diag(n,n)`. The following complete example compares
two designs against a prespecified threshold. The Hessian of the Gaussian
negative log likelihood gives this reference independently of the screen.

```python
import numpy as np
from trade_study import (
    Direction, FisherContext, FisherScreenPolicy, Observable, ResultsTable,
    fisher_screen_counts, screen_fisher,
)

def context(n):
    return FisherContext(
        parameter_names=("a", "b"),
        parameter_coordinates="physical a,b; dimensionless",
        parameter_scales=(1.0, 1.0),
        observation_model="independent Gaussian, known variance=1",
        design=f"{n} balanced measurements at x=-1,+1",
        reference_regime="linear mean; all a,b",
        nuisance_treatment="none; both coefficients targeted",
        sensitivity_reference="analytic d(a+b*x)/d(a,b)=(1,x)",
        numerical_reference="exact design; host float64 algebra",
        qualification_reference="analytic Gaussian likelihood Hessian",
        information_reference=f"retained synthetic design n={n}",
    )

policy = FisherScreenPolicy(threshold=4.0)
records = {
    str(n): screen_fisher(np.eye(2) * n, context=context(n), policy=policy)
    for n in (2, 8)
}
table = ResultsTable(
    configs=[{"n": n} for n in (2, 8)],
    scores=np.array([[records[str(n)].as_scores()["fisher_min_eigenvalue"]]
                     for n in (2, 8)]),
    observable_names=["fisher_min_eigenvalue"],
    metadata=[{"fisher_screen": records[str(n)].to_metadata()} for n in (2, 8)],
)
observable = Observable("fisher_min_eigenvalue", Direction.MAXIMIZE)
assert table.feasible([policy.constraint()]).tolist() == [False, True]

# Each unavailable attempt keeps its identity and reason.
records["missing-artifact"] = screen_fisher(None, context=context(8), policy=policy)
records["failed-solve"] = screen_fisher(
    None, context=context(8), policy=policy, failure="upstream solve failed"
)
counts = fisher_screen_counts(
    records, planned_ids=["2", "8", "missing-artifact", "failed-solve", "pending"]
)
assert counts == dict(planned=5, feasible=1, infeasible=1,
                      unreadable=1, failed=1, missing=1)
```

`to_metadata()` returns detached primitive JSON for existing result bundles.
Retain the per-attempt records and complete planned IDs alongside any scored
table. The two numeric rows above are not the denominator for all five planned
attempts. `as_scores()` raises for unreadable/failed outcomes, so the generic
runner follows its existing failure/checkpoint policy rather than silently
averaging them away. These outcome counts alone supply no probability or gate.

## Coordinates, targets and numerical limits

The input matrix is already expressed in its declared coordinates and scales.
This adapter never rescales it automatically. Thresholds have meaning in those
coordinates; a change of units requires a correspondingly qualified policy.
Context references declare the observation law/noise, design, parameter regime,
nuisance convention and sensitivity/numerical/qualification evidence. The caller
retains and verifies those artifacts. The recorded matrix digest identifies the
actual float64 host snapshot, independently of the caller's reference labels.

An optional p-by-k `directions` basis has orthonormal columns, checked with
absolute and relative tolerances of `1e-10`. The tested matrix is `B.T @ F @ B`,
with the complementary directions fixed. This is useful for explicitly declared
identifiable perturbations of a singular nominal matrix. For observations of
`a+b`, the normalized `(1,1)` direction has information while `(1,-1)` does not.
The caller must justify which targets and gauge directions this represents.

Directional restriction does not perform nuisance elimination. For example,
`F=[[1,1],[1,2]]` has conditional information 1 for its first coordinate, while
profiling the second coordinate gives `1 - 1*(1/2)*1 = 0.5`. Supply the caller's
profiled matrix and its distinct context when that is the intended calculation.

The default dense dimension limit is 128 parameters. Inputs are real NumPy host
arrays; a device-to-host transfer is an explicit caller boundary. Matrix and
eigensolver storage remain quadratic in dimension; the limit does not bound
whole-process or AD/compiler memory. No sensitivity array is retained here.

Invalid shape/dtype, nonfinite, asymmetric, indefinite or unsupported matrices
produce unreadable records. Upstream/eigensolver failures produce failed records.
Symmetry and positive-semidefiniteness use
`atol + rtol * max(abs(F))`. Accepted near-symmetric matrices are averaged with
their transpose. Actual tested eigenvalues are retained; tiny negative values
are not clipped and comparison with the threshold is inclusive and exact on the
computed value. Numerical tolerances do not relax the decision threshold.

## Qualification beyond a screen

A local information screen helps shortlist designs. A design decision also needs
declared simulation/refitting targets, numerical checks and relevant recovery
and predictive evidence. In particular:

- SBC tests the selected posterior computations under a declared generator and
  design. Finite non-rejection supports “no miscalibration detected”. Ordinary
  parameter ranks can miss data-ignoring estimators; appropriately chosen
  data-dependent quantities improve sensitivity ([Modrák et al.](https://arxiv.org/abs/2211.02383)).
- Effect recovery uses the actual reporting rule and design. Define signed
  effects in a null-centred coordinate. Type S and Type M condition on reporting;
  zero decisions leave them unavailable. At zero effect, report false-support
  rather than an undefined magnitude ratio. Preserve failures and conditional
  denominators ([Gelman and Carlin](https://sites.stat.columbia.edu/gelman/research/published/retropower_final.pdf)).
- Predictive checks examine the relevant observed features and held-out units.
  Ordinary in-sample posterior predictive probabilities need not be uniform.
  Randomized ties address atoms; a rejection gate still needs a qualified
  reference distribution and dependence/multiplicity treatment
  ([Gelman](https://sites.stat.columbia.edu/gelman/research/published/ppc_understand3.pdf)).
- Amortized estimators also need checks of the observed data against their
  training-simulation distribution. A simulator-generated SBC bank does not
  directly test that observed-data discrepancy.

Repeated information matrices estimate a probability only under a declared
sampling law. A confidence argument on a deterministic matrix creates no such
law. Use [fixed-sample proportions](proportion-intervals.md) for their supported
Bernoulli assumptions, or the [sequential runner](../api/runner.md) for its
bounded-score allocation contract. Keep planned attempts, failed and missing
results, independent/cluster units and the stopping rule explicit. Interval
precision, gate power and false-failure rates each require their own evidence.

General Fisher families, nuisance profiling, global structural-identifiability
analysis, posterior adequacy and EVSI retain their caller/inference-layer scope.
