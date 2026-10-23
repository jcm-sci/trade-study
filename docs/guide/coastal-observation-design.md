# Coastal observation design with fixed scenario banks

This example compares 27 hypothetical coastal observation designs. It shows
how to keep null and event simulations separate, share random-number banks
across designs, summarize replicated detections with Wilson intervals, apply a
false-alarm rule, and extract a three-objective Pareto front.

!!! warning "Teaching inputs only"

    Every signal, measurement-error scale, threshold, cost, and decision rule
    in this example is invented. The values are not field measurements,
    instrument specifications, facility quotes, partner commitments, protocol
    choices, or recommendations for a real monitoring program.

The complete script is
[`examples/coastal_observation_design.py`](https://github.com/jcm-sci/trade-study/blob/main/examples/coastal_observation_design.py).
The fast check runs the complete workflow with 60 Monte Carlo trials per design
and scenario on one CPU process:

```bash
uv run --extra pareto python examples/coastal_observation_design.py --check
```

Omit `--check` to use the example's prespecified 400 trials per design and
scenario. The `--n-jobs` option changes CPU parallelism for that full run;
values other than one require the `parallel` extra, while `--check` always uses
one process.

## Decision question and quantities

The teaching question is whether additional sampling events, technical
replicates, or a different measurement mode improve event detection and
intensity estimation enough to justify their hypothetical cost. The latent
event intensity and measurement errors use arbitrary intensity units. Cost is
reported in invented cost units, without a currency conversion.

Each design has three factors:

| Factor | Levels | Meaning |
| --- | --- | --- |
| Measurement mode | rapid sensor, reference assay, paired cross-platform | Three invented error and cost settings |
| Sampling events | 4, 7, 13 | Fixed subsets of 13 times in one normalized monitoring window |
| Technical replicates | 1, 2, 3 | Nested measurements averaged at each sampling time |

The full factorial grid therefore contains $3 \times 3 \times 3 = 27$
designs. The synthetic scorer reports a binary detection indicator, squared
intensity error, and deterministic cost. Event intensity root mean squared
error (RMSE) is the square root of mean squared error across event-bank trials.
For each trial, estimated intensity is the nonnegative maximum of the sampled
values, and detection is one when that estimate is at least 0.80. The mean
detection indicator estimates detection probability in the event bank and
false-alarm probability in the null bank. Event-bank RMSE compares estimated
intensity with the latent event amplitude.

```python
--8<-- "examples/coastal_observation_design.py:design"
```

## Prespecify the null and event banks

The null bank fixes the latent event amplitude at zero. The event bank draws a
positive Gaussian-shaped event with a random amplitude, center, and width. A
bank contains `n_reps` independent Monte Carlo trials indexed from zero through
`n_reps - 1`; the bank names and root seed are fixed before evaluating designs.

```python
--8<-- "examples/coastal_observation_design.py:scenario-banks"
```

Within a bank, the same trial index selects the same latent event, calibration
draw, and standardized measurement-error array for every design. This is a
common-random-number construction: designs are compared on matched synthetic
conditions, while their measurement mode, sampling schedule, and technical
replication transform or subset the shared draws. Common random numbers can
reduce Monte Carlo noise in design contrasts, but they do not add independent
trials or validate the assumed data-generating process.

```python
--8<-- "examples/coastal_observation_design.py:world"
```

## Aggregate detections and intensity errors

For each bank, `run_grid(..., n_reps=n_reps)` retains every design and trial
identity. `ResultsTable.aggregate_replicates` then computes the mean of each
outcome. Passing `proportion_observables=["detected"]` also records the number
of detections, number of trials, observed proportion, and a two-sided marginal
95% Wilson interval without continuity correction.

The Wilson intervals describe fixed-sample Monte Carlo uncertainty under the
example's independent-trial construction. They are marginal intervals: the
example does not adjust them for examining 27 designs, using the null intervals
for selection, or reporting the selected Pareto front. See
[Fixed-sample proportions](proportion-intervals.md) for the API contract and
precision-planning boundaries.

## Apply the false-alarm rule before Pareto extraction

The example calls a design feasible when the upper endpoint of its null-bank
95% Wilson interval is no greater than the hypothetical false-alarm limit of
0.10. Equality passes. This conservative teaching rule uses the entire fixed
null bank and is not a regulatory, ecological, or operational standard.

Among feasible evaluated designs, `extract_front` uses three objectives:

| Objective | Direction | Scenario bank |
| --- | --- | --- |
| Detection probability | Maximize | Event |
| Intensity RMSE | Minimize | Event |
| Invented cost units | Minimize | Deterministic, identical in both banks |

```python
--8<-- "examples/coastal_observation_design.py:run"
```

The returned flag `pareto_among_evaluated_feasible` has a deliberately narrow
meaning. A flagged design is nondominated only among the feasible members of
this 27-design grid under the finite synthetic banks. The front is not a global
optimum and does not identify a preferred real-world design.

## Limitations

- The event shape, Gaussian errors, threshold, and mode parameters are
  hypothetical; real observations may have temporal, spatial, calibration,
  censoring, failure, and distributional structure absent here.
- The peak-based intensity estimator can be positively biased because it takes
  a maximum over noisy observations. More sampling events can change both this
  bias and the detection probability.
- The Wilson intervals are approximate, marginal, and fixed-sample. They do not
  supply simultaneous coverage, a post-selection guarantee, or permission to
  stop simulation when a desired result appears.
- The same finite banks support feasibility screening and Pareto extraction.
  An independent validation bank would be needed to assess a selected design's
  operating characteristics without this reuse.
- Common random numbers support matched comparisons and can improve their
  precision when the coupling induces favorable covariance. They do not
  increase the number of independent event or null trials.
- Invented cost units omit procurement, staffing, logistics, maintenance,
  failures, data management, permitting, and other program-specific costs.
