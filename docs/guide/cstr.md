# Reactor Design Study

This tutorial walks through a **multi-phase trade study** for a
continuous stirred-tank reactor (CSTR) with competing objectives.
The reactor model uses closed-form Arrhenius kinetics — no dependencies
beyond numpy.

!!! tip "Run it yourself"

    The full runnable script is at
    [`examples/cstr_study.py`](https://github.com/jcm-sci/trade-study/blob/main/examples/cstr_study.py).

    ```bash
    uv run python examples/cstr_study.py
    ```

## The problem

A **continuous stirred-tank reactor** (CSTR) is a standard unit
operation in chemical engineering.  Reactants flow in continuously, mix
perfectly inside the vessel, and products flow out at steady state.

We model two series reactions:

$$A \xrightarrow{k_1} B \xrightarrow{k_2} C$$

where **B** is the desired product and **C** is an unwanted byproduct.
Each rate constant follows the Arrhenius law:

$$k_i = A_i \exp\!\left(-\frac{E_{a,i}}{R\,T}\right)$$

At steady state, the CSTR mass balance gives the exit concentrations:

$$C_A = \frac{C_{A,\text{in}}}{1 + k_1 \tau}, \qquad
  C_B = \frac{k_1 \tau \, C_{A,\text{in}}}{(1 + k_1 \tau)(1 + k_2 \tau)}$$

where $\tau$ is the mean residence time. From these we define:

- **Conversion**: $X = 1 - C_A / C_{A,\text{in}}$ — fraction of A
  consumed (higher is better).
- **Selectivity**: $S = C_B / (C_{A,\text{in}} - C_A)$ — fraction of
  converted A that became the *desired* product B (higher is better).
- **Energy cost**: $0.01(T - 300)^2 + 5 \dot{V}_c$ — a simplified
  model of heating plus coolant pumping costs (lower is better).

### Why the objectives conflict

Raising the reactor temperature increases $k_1$ (better conversion),
but also accelerates the unwanted $B \to C$ reaction because $E_{a,2}
> E_{a,1}$, which *hurts* selectivity.  Meanwhile, both heating and
cooling cost energy.  There is no single set of operating conditions
that simultaneously maximizes conversion and selectivity while
minimizing cost — the solutions lie on a **Pareto front**.

## Ground-truth model

The code below implements the steady-state equations as a pure Python
function.  Because the model is closed-form, every evaluation takes
microseconds — ideal for demonstrating `trade-study` without waiting
for expensive simulations.

```python
;--8<-- "examples/cstr_study.py:kinetics"
```

## Simulator and scorer

`trade-study` uses two protocols:

- A **Simulator** produces `(truth, observations)` pairs — here, the
  truth is the exact steady-state output and the observations add 2 %
  Gaussian noise to simulate real measurement uncertainty.
- A **Scorer** extracts the objective values that will populate the
  results table.

```python
;--8<-- "examples/cstr_study.py:world"
```

## Define observables and factors

**Observables** declare *what* we measure and which direction is better.
Each one becomes a column in the results table.

```python
;--8<-- "examples/cstr_study.py:observables"
```

**Factors** declare *what we can control* — here, four continuous
operating parameters with physical bounds:

```python
;--8<-- "examples/cstr_study.py:factors"
```

## Build the study phases

A `Study` chains multiple **Phases**.  Phase 1 explores the design space
broadly with a 60-point Latin hypercube.  The `top_k_pareto_filter(20)`
callback selects the 20 best designs (by Pareto rank) and passes them to
Phase 2.  Phase 2 re-evaluates those 20 designs for confirmation (with
fresh noise draws), acting as a validation step.

```python
;--8<-- "examples/cstr_study.py:phases"
```

## Run and inspect results

Create a `Study`, call `.run()`, and then query results per phase:

```python
;--8<-- "examples/cstr_study.py:run"
```

The results include the Pareto front, per-design scores, and the
**hypervolume indicator** — a single number summarizing how well the
front covers the objective space (higher is better):

```python
;--8<-- "examples/cstr_study.py:results"
```

## What to try next

- Add a **screening** step with `screen()` to identify which factors
  matter most before running the full grid.
- Use `run_adaptive()` for Bayesian optimization instead of LHS.
- Call `stack_bayesian()` on the Pareto front to compute model-averaging
  weights.
- Save results with `save_results()` for later analysis.
