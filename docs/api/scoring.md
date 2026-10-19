# Scoring

Proper scoring rules for posterior predictive evaluation.

CRPS and coverage treat each row as a univariate case: predictions are
`(*cases, members)` and truth is `(*cases,)`. Energy instead treats each case as
a multivariate vector: predictions are `(*cases, members, variables)` and truth
is `(*cases, variables)`. A single jointly scored vector therefore uses a
`(members, variables)` ensemble. The Bayesian tutorial stores predictions as
`(test_location, member)` for marginal CRPS/coverage and transposes them for
joint energy scoring. RMSE/MAE compare aligned point predictions with truth.

::: trade_study.score

::: trade_study.coverage_curve
