# Design

Experimental design: factors, grids, and screening.

## Sobol evaluation budgets

For base sample size `N` and `D` continuous factors, both
`screen(method="sobol", n_trajectories=N)` and `sobol_indices(n_samples=N)`
currently request the second-order Saltelli design with `N * (2*D + 2)` nominal
configurations. `screen` returns first-order indices, but still uses that larger
design. A completed run invokes `run_fn` `N * (2*D + 2) * n_reps` times.
Repeated configurations in the design do not reduce that callback count.

For example, `N=8`, `D=3` and `n_reps=2` mean 64 nominal configurations and
128 callable invocations. Replication costs apply even when a callback does not
accept the optional `rep` keyword; meaningful stochastic replication requires
the caller's randomness contract. These counts describe the screening callback,
not hidden work or caching inside it. Surrogate fitting/CV has its own cost.
This correction preserves the sampler/analyzer settings and existing defaults;
there is no new first/total-only sampling option.

::: trade_study.FactorType

::: trade_study.Factor

::: trade_study.FactorConstraint

::: trade_study.build_grid

::: trade_study.reduce_factors

::: trade_study.screen

::: trade_study.sobol_indices

::: trade_study.unit_to_value

::: trade_study.value_to_unit
