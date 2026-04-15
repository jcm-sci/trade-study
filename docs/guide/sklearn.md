# Hyperparameter Trade Study

This tutorial runs a **multi-objective hyperparameter sweep** over
scikit-learn's `GradientBoostingRegressor`, balancing prediction
accuracy against training cost and model complexity.

!!! tip "Run it yourself"

    The full runnable script is at
    [`examples/sklearn_study.py`](https://github.com/jcm-sci/trade-study/blob/main/examples/sklearn_study.py).

    ```bash
    pip install scikit-learn
    uv run python examples/sklearn_study.py
    ```

## The problem

Machine learning practitioners routinely tune hyperparameters to
minimize prediction error, but in production **accuracy is not the only
objective**.  A model that takes 10× longer to train or has 100×
more parameters may not be worth the marginal RMSE improvement.

This example treats hyperparameter selection as a **multi-objective
design-of-experiments** problem with three competing objectives:

| Objective | Direction | What it measures |
|-----------|-----------|------------------|
| RMSE | minimize | Root mean squared error on a held-out test set |
| Training time | minimize | Wall-clock seconds to `fit()` the model |
| Complexity | minimize | Total number of leaf nodes across all trees |

### Why the objectives conflict

- **More estimators** → lower RMSE, but longer training and more leaves.
- **Deeper trees** → lower RMSE, but each tree has exponentially more
  leaves and takes longer to build.
- **Higher learning rate** → faster convergence with fewer trees, but
  risks overfitting if not paired with regularization.
- **Lower subsample** → implicit regularization (less overfitting),
  but noisier gradient estimates.

No single hyperparameter setting wins on all three objectives —
the solutions lie on a **Pareto front**.

## Dataset

We use scikit-learn's `make_friedman1`, a standard synthetic regression
benchmark.  The true function is:

$$y = 10\sin(\pi x_1 x_2) + 20(x_3 - 0.5)^2 + 10 x_4 + 5 x_5 + \varepsilon$$

where $\varepsilon \sim \mathcal{N}(0, 1)$ and $x_6, \dots, x_{10}$
are noise features.  We generate 800 samples and hold out 25 % for
testing:

```python
;--8<-- "examples/sklearn_study.py:dataset"
```

## Simulator and scorer

In `trade-study`, every experiment needs a **Simulator** and a
**Scorer**:

- The **Simulator** wraps model training.  Its `generate()` method
  receives a hyperparameter config, fits a `GradientBoostingRegressor`,
  and returns the test-set ground truth plus predictions and metadata
  (training time, leaf count).
- The **Scorer** extracts the three objective values from each trial.

```python
;--8<-- "examples/sklearn_study.py:world"
```

## Define observables and factors

**Observables** tell `trade-study` *what* you're measuring and which
direction is better:

```python
;--8<-- "examples/sklearn_study.py:observables"
```

**Factors** define the hyperparameter search space.  We use discrete
levels so a full factorial grid is tractable
($4 \times 4 \times 4 \times 3 = 192$ combinations):

```python
;--8<-- "examples/sklearn_study.py:factors"
```

## Run the sweep

`build_grid` with `method="full"` generates every combination.
`run_grid` evaluates them all and returns a `ResultsTable`:

```python
;--8<-- "examples/sklearn_study.py:run"
```

## Inspect the Pareto front

`extract_front` identifies the subset of designs where **no other
design is better on all objectives simultaneously**.  Walking along
the front reveals the fundamental trade-off: lower RMSE costs more
training time and model complexity.

```python
;--8<-- "examples/sklearn_study.py:results"
```

The Pareto front typically contains 10–15 designs out of 192.
A practitioner can then choose based on their priorities — e.g. pick
the lowest-RMSE design if accuracy is paramount, or the fastest design
that still meets an RMSE threshold.

## What to try next

- Use `build_grid(factors, method="lhs", n_samples=50)` for a Latin
  hypercube design with continuous hyperparameters.
- Wrap the sweep in a `Study` with multiple phases — screen first,
  then refine the promising region.
- Add an `Annotation` for dollar cost (e.g., cloud compute pricing per
  second) to include cost as a non-simulated objective.
- Use `save_results()` / `load_results()` to persist results across
  sessions.
