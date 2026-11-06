# Power Curves

Simulation-based power analysis reads a raw 0/1 indicator observable, such as
"the simulated trial rejected the null", from replicated `run_grid` rows.
`power_curve` orders the per-design success probability along one size factor,
separately for every combination of the other factors. `minimum_adequate` then
reports the smallest scanned size whose power meets a target.

```python
curves = power_curve(raw_results, "reject", "n_participants", adjustment="bonferroni")
for choice in minimum_adequate(curves, 0.8):
    print(choice.stratum, choice.value, choice.monotone_above)
```

Each interval is a fixed-sample
[proportion interval](proportions.md) for independent replicates; the caller
establishes common, independent replication. Without `adjustment`, the bound
at the selected size is marginal. Choosing the first crossing from a scanned
curve is a selection, so `adjustment="bonferroni"` widens each interval to keep
the nominal level simultaneously across the stratum's scanned values.
`monotone_above=False` reports a curve that falls back below the target at a
larger size; curves are never smoothed or interpolated, and the result is only
as precise as the scanned grid.

::: trade_study.PowerCurve

::: trade_study.AdequateDesign

::: trade_study.power_curve

::: trade_study.minimum_adequate
