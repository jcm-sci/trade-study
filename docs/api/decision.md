# Decision summaries

Use existing results to see how choices change with preferences:

```python
policy = PreferencePolicy(
    normalization="minmax",
    weights=[
        {"accuracy": weight, "cost": 1 - weight}
        for weight in (0.25, 0.5, 0.75)
    ],
    equivalence={"accuracy": 0.01, "cost": 5.0},
)
sweep = preference_sweep(results, observables, policy=policy, constraints=constraints)
frame = sweep.summary.to_dataframe(include_metadata=True)
```

Every preference vector is explicit, finite and nonnegative and is normalized
by its sum; omitted objectives have zero weight. `Observable.weight` is
replaced by these preference vectors. Minimized objectives contribute positively
and maximized objectives negatively to the weighted loss; smaller loss wins.
Session raw means in metadata override already-weighted objective columns.
Raw replicated tables are aggregated by design point, with duplicate or
incomplete replicate identities refused. The summary retains raw-unit means.

Choose normalization explicitly. `minmax` uses the finite feasible alternatives
in this table, so changing that set can change rankings. `reference` uses
caller-provided raw-unit `reference_bounds` and allows values beyond the anchors
without clipping. `none` retains original units, making the weight ratios unit
dependent. Zero-range objectives contribute equally to every eligible design.
Outputs expose the selected policy, effective weights and actual anchors.

`ranks` shows each design's competition rank across preferences; ties share a
rank. Summary metadata includes the best/worst/mean rank, maximum weighted-loss
regret and selection fraction. Tied winners split selection credit, so fractions
sum to one when eligible alternatives exist. These are fractions of the supplied
preference scenarios, not posterior probabilities or sampling confidence.
`pareto` identifies feasible nondominated alternatives regardless of preferences.
Infeasible/non-finite designs have NaN ranks and zero selection credit; if none
remain, the result reports no choice. Feasibility follows existing constraint
rules, including required standard errors for confidence constraints.

`equivalent[a, b]` compares every objective's raw mean difference with the
specified practical tolerance; omitted tolerances are zero. This is a pairwise
relation and need not be transitive. It expresses practical similarity of point
estimates, not statistical evidence that the designs are equivalent.

For raw data with matching replicate sets, supply `paired_reference` as a design
point id or config selector to attach the existing paired comparisons. Only
eligible designs are compared. The requested `paired_confidence` is split across
observables and the existing Bonferroni adjustment across alternatives. Bootstrap
coverage remains approximate; paired t assumptions remain unchanged. Choosing a
reference after observing outcomes and optional stopping are not corrected.
Unequal replication needs an explicitly chosen common replicate set before
calling this API. Without raw replicate identities, paired uncertainty cannot
be reconstructed from means and standard errors alone.

::: trade_study.PreferencePolicy

::: trade_study.PreferenceSweep

::: trade_study.preference_sweep
