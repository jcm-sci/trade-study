# Reactor Design Study

This tutorial walks through a **multi-phase trade study** for a
continuous stirred-tank reactor (CSTR) with competing objectives.
The reactor model uses closed-form Arrhenius kinetics — no dependencies
beyond numpy.

The full runnable script is at
[`examples/cstr_study.py`](https://github.com/jcm-sci/trade-study/blob/main/examples/cstr_study.py).

## Problem setup

A CSTR converts feedstock A → B → C in series reactions.
**B** is the desired product. We want to find operating conditions that
balance three competing objectives:

| Objective | Direction | Why it conflicts |
|-----------|-----------|------------------|
| Conversion of A | maximize | Higher temperature helps, but… |
| Selectivity for B | maximize | …high temperature also accelerates B → C |
| Energy cost | minimize | Heating and cooling both cost money |

## Ground truth model

The reactor model computes steady-state concentrations from
Arrhenius rate constants and a residence-time balance:

```python
--8 < --"examples/cstr_study.py:kinetics"
```

## Simulator and scorer

`trade-study` uses two protocols: a **Simulator** that generates
(truth, observations) pairs, and a **Scorer** that extracts objective
values. Here the simulator adds 2% Gaussian noise to simulate
measurement uncertainty:

```python
--8 < --"examples/cstr_study.py:world"
```

## Define observables and factors

Observables declare *what* we measure and which direction is better.
Factors declare *what* we can control:

```python
--8 < --"examples/cstr_study.py:observables"
```

```python
--8 < --"examples/cstr_study.py:factors"
```

## Build the study phases

Phase 1 explores broadly with a Latin hypercube design (60 points).
The `top_k_pareto_filter(20)` passes the 20 best designs to Phase 2,
which re-evaluates them for confirmation:

```python
--8 < --"examples/cstr_study.py:phases"
```

## Run and inspect results

```python
--8 < --"examples/cstr_study.py:run"
```

```python
--8 < --"examples/cstr_study.py:results"
```

## What to try next

- Add a **screening** step with `screen()` to identify which factors
  matter most before running the full grid.
- Use `run_adaptive()` for Bayesian optimization instead of LHS.
- Call `stack_bayesian()` on the Pareto front to compute model-averaging
  weights.
- Save results with `save_results()` for later analysis.
