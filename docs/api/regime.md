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

::: trade_study.RegimeSurrogate
