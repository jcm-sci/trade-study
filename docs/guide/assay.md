# Assay choices with an external cost table

This synthetic example compares categorical assay labels and continuous
classification thresholds, maximizing accuracy and minimizing unit cost.
All costs and response distributions are invented; the assay labels are
illustrative and the simulation does not represent measured clinical performance.

Run the complete example from the repository:

```bash
uv run --extra examples python examples/assay_study.py
```

The script is
[`examples/assay_study.py`](https://github.com/jcm-sci/trade-study/blob/main/examples/assay_study.py).

## Attach external costs

A lookup table associates each categorical assay with a unit cost:

```python
UNIT_COST = {"PCR": 18.0, "ELISA": 8.0, "rapid": 3.0}
annotations = [
    Annotation(name="unit_cost", lookup=UNIT_COST, key="assay")
]
```

For `{"assay": "ELISA", "threshold": 0.5}`, `Annotation.resolve()` reads the
`assay` value and looks up `8.0`. Annotations populate a separate matrix alongside
the score columns. They can record external costs or other information without
changing what the simulator generates. Using a distinct annotation name also
keeps DataFrame exports clear.

## Make cost part of the decision

An annotation alone does not add a Pareto objective. To optimize cost, the scorer
returns a `cost` score from the same table and the study declares its direction:

```python
observables = [
    Observable("accuracy", Direction.MAXIMIZE),
    Observable("cost", Direction.MINIMIZE),
]

# Inside AssayScorer.score(...):
return {
    "accuracy": float(np.mean(truth == observations)),
    "cost": UNIT_COST[config["assay"]],
}
```

`cost` participates in Pareto sorting, while `unit_cost` remains an externally
resolved annotation. Sharing the table avoids maintaining two conflicting cost
sources. These are per-assay costs; if total study expense is the objective,
compute that quantity explicitly in the scorer instead.

## Screen, then refine retained designs

The grid combines three assay labels with three initial thresholds. The
categorical factor describes the labels; the threshold factor declares a
continuous range, even though this initial grid uses three explicit points.

The screening phase evaluates each design with three replicates of 300 synthetic
cases. `top_k_pareto_filter(4)` retains four designs using their mean objectives.
The refinement phase uses `grid="carry"` and a larger simulator with 3,000 cases
per replicate, reducing Monte Carlo variability while keeping both objectives
and the cost annotation.

```python
phases = [
    Phase("screen", grid=grid, filter_fn=top_k_pareto_filter(4), n_reps=3),
    Phase("refine", grid="carry", world=AssayWorld(n_cases=3000), n_reps=3),
]
```

The simulator accepts `rep` and uses `seed + rep`. Within a phase, matching
replicate ids share the synthetic case/noise stream across assay and threshold
choices. The scorer measures classification accuracy against those synthetic
truth labels; the external cost stays fixed for an assay.

## Inspect annotations and means

Raw phase results retain one row per replicate. Aggregate them for a per-design
report and access the annotation matrix separately:

```python
results = study.results("refine").aggregate_replicates()
print(results.observable_names)  # ["accuracy", "cost"]
print(results.annotation_names)  # ["unit_cost"]
print(results.annotations)       # one unit-cost value per design
```

The runnable script prints each retained assay/threshold, mean accuracy, cost
objective and unit-cost annotation. The cost values in both paths agree, while
accuracy varies with the simulated readouts and threshold. Retained designs are
trade-off alternatives; the front does not specify a single final preference.

## Complete source

```python
--8<-- "examples/assay_study.py"
```
