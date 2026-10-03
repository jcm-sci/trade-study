# Surrogate

Cheap regression surrogates fit over a [`ResultsTable`][trade_study.ResultsTable]
for predicting observables at untested configurations.

Install via the optional extra:

```bash
uv pip install 'trade-study[surrogate]'
```

::: trade_study.fit_surrogate

## Validation and support

`fit_surrogate(..., cv_group_by="design")` keeps identical factor configurations
in one validation fold, preventing replicates of a design from appearing in
both training and validation. Pass a sequence such as `cv_group_by=["noise"]`
to hold out whole regimes. `cv_r2` and `cv_rmse` then describe held-out groups;
`row_cv_r2` and `row_cv_rmse` retain shuffled row validation for comparison.
The default `cv_group_by=None` retains the existing row validation behavior.
Grouping is explicit: different design-point ids with identical factor values
belong to the same design group. Fewer than two groups produce `nan` metrics;
constant targets have undefined R². Non-finite scores are excluded separately
for each observable. `validation_groups` exposes row-aligned group ids.

`model.support(configs)` reports continuous factors outside the observed
training range and categorical/discrete levels absent from finite training rows
for each observable. Predictions warn by default and still extrapolate without
clipping. Use `warn_support=False` to disable these warnings explicitly.
Levels outside the factor's declared domain remain invalid. Bounds use raw
factor units, including for log-scaled factors. Marginal ranges do not establish
joint support, adequate sampling density, or calibrated prediction uncertainty.

::: trade_study.SurrogateModel

::: trade_study.PredictionSupport
