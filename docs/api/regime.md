# Regime Surrogate

Regime-conditional surrogate that interpolates factor recommendations
across regime descriptors (e.g. dataset size, noise level) instead of
relying on hard regime buckets. Builds on
[`fit_surrogate`][trade_study.fit_surrogate].

Install via the optional extra (same as the base surrogate):

```bash
uv pip install 'trade-study[surrogate]'
```

::: trade_study.fit_regime_surrogate

`fit_regime_surrogate(..., cv_group_by="regime")` holds out entire tuples of
regime descriptors. The underlying model exposes `cv_group_by` and group ids;
`cv_r2`/`cv_rmse` describe held-out regimes, while `row_cv_r2`/`row_cv_rmse`
describe shuffled rows. Use `"design"` for joint design/regime configurations
or an explicit sequence of factor names for another grouping.

Inspect `model.support(regime, configs)` before trusting a recommendation.
Prediction and recommendation warn when candidates or the query regime fall
outside observed marginal support. `warn_support=False` disables that warning;
`warn_below_r2=None` independently disables accuracy warnings.

::: trade_study.RegimeSurrogate
