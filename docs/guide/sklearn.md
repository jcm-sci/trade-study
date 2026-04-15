# Hyperparameter Trade Study

This tutorial runs a **multi-objective hyperparameter sweep** over
scikit-learn's `GradientBoostingRegressor`, balancing prediction
accuracy against training cost and model complexity.

The full runnable script is at
[`examples/sklearn_study.py`](https://github.com/jcm-sci/trade-study/blob/main/examples/sklearn_study.py).

## Problem setup

We train gradient boosting models on the synthetic Friedman #1 dataset
and evaluate three objectives:

| Objective | Direction | Meaning |
|-----------|-----------|---------|
| RMSE | minimize | Prediction error on the test set |
| Training time | minimize | Wall-clock seconds to fit |
| Complexity | minimize | Total number of leaves across all trees |

These objectives compete: more estimators and deeper trees reduce RMSE
but increase training time and complexity.

## Dataset

```python
--8 < --"examples/sklearn_study.py:dataset"
```

## Simulator and scorer

The simulator trains a model and returns test-set predictions plus
metadata. The scorer extracts the three objectives:

```python
--8 < --"examples/sklearn_study.py:world"
```

## Define observables and factors

```python
--8 < --"examples/sklearn_study.py:observables"
```

```python
--8 < --"examples/sklearn_study.py:factors"
```

Four factors with 4 × 4 × 4 × 3 = 192 combinations in a full
factorial design.

## Run the sweep

```python
--8 < --"examples/sklearn_study.py:run"
```

## Inspect the Pareto front

```python
--8 < --"examples/sklearn_study.py:results"
```

The Pareto front shows the set of designs where **no single design
dominates another on all three objectives**. Moving along the front
reveals the trade-off: lower RMSE costs more training time and model
complexity.

## What to try next

- Use `build_grid(factors, method="lhs", n_samples=50)` for a Latin
  hypercube design with continuous hyperparameters.
- Wrap the sweep in a `Study` with multiple phases — screen first,
  then refine the promising region.
- Add an `Annotation` for dollar cost (e.g., cloud compute pricing per
  second) to include cost as a non-simulated objective.
- Use `save_results()` / `load_results()` to persist results across
  sessions.
