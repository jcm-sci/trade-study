# Monitoring Station Design

This tutorial runs a **multi-objective trade study** over instrument
settings for a remote monitoring station.  The goal is to recover a
known environmental signal as faithfully as possible while keeping
station cost low.

Unlike the CSTR and sklearn examples — where the experiment changes
*what* is built or trained — here the experiment changes *how you
observe*.  The underlying signal is fixed; only the instrument
configuration varies.

!!! tip "Run it yourself"

    The full runnable script is at
    [`examples/monitoring_study.py`](https://github.com/jcm-sci/trade-study/blob/main/examples/monitoring_study.py).

    ```bash
    uv run python examples/monitoring_study.py
    ```

## The problem

A reference environmental signal is measured at a remote station.
The instrument settings — sample rate, ADC resolution, sensor quality,
and number of co-located sensors — determine how faithfully the true
signal is recovered and how much the station costs.

### Reference signal

A synthetic 24-hour signal with three components (numpy only):

- **Diurnal cycle**: smooth sinusoid with a 24-hour period.
- **Short-period oscillations**: two higher-frequency modes at periods
  of 5 minutes and 20 minutes (e.g. tidal harmonics, turbulence).
- **Correlated noise**: an AR(1) process ($\varphi = 0.995$,
  decorrelation time $\approx 200$ s) representing natural variability.

The signal is generated at 1-second resolution — $N = 86\,400$ samples
for a full day.

### Observation model

The instrument applies a chain of realistic transformations:

1. **Anti-alias filter** — brick-wall low-pass at the Nyquist frequency
   of the configured sample rate.
2. **Decimation** — subsample to the configured interval.
3. **Quantization** — finite ADC resolution maps continuous values to
   $2^b$ discrete levels.
4. **Sensor noise** — additive Gaussian noise whose standard deviation
   depends on sensor grade, reduced by $\sqrt{n}$ averaging when
   multiple co-located sensors are used.

### Why the objectives conflict

- **Faster sampling + more bits + better sensors + more averaging** →
  near-perfect recovery, but station cost explodes.
- **Cheap configurations** (300 s interval, 8-bit, field-grade,
  1 sensor) lose the short-period oscillations entirely and quantize
  the diurnal cycle into visible steps.
- **Mid-range configs** face a genuine Pareto trade-off: you can spend
  your budget on temporal resolution *or* sensor quality, but rarely
  both.

## Reference signal generation

```python
--8<-- "examples/monitoring_study.py:signal"
```

## Observation model

The `observe()` function implements the four-stage signal chain
described above.  Each stage is a few lines of numpy:

```python
--8<-- "examples/monitoring_study.py:observation"
```

## Simulator and scorer

The **Simulator** applies the observation model with a per-config
deterministic seed.  The **Scorer** computes three objectives:

| Observable | Direction | What it measures |
|------------|-----------|------------------|
| RMSE | minimize | RMS error after interpolating back to the reference grid |
| Spectral fidelity | maximize | Fraction of spectral power recovered in the 10 min – 2 h band |
| Station cost | minimize | Additive cost model over sample rate, ADC, grade, and sensor count |

```python
--8<-- "examples/monitoring_study.py:world"
```

## Define observables and factors

**Observables** declare what we measure and which direction is better:

```python
--8<-- "examples/monitoring_study.py:observables"
```

**Factors** define the instrument search space.  The full factorial
grid is $5 \times 4 \times 3 \times 4 = 240$ configurations:

```python
--8<-- "examples/monitoring_study.py:factors"
```

## Run the sweep

`build_grid` with `method="full"` generates every combination.
`run_grid` evaluates them all and returns a `ResultsTable`:

```python
--8<-- "examples/monitoring_study.py:run"
```

## Inspect the Pareto front

The Pareto front identifies configurations where no other design
dominates on all three objectives.  Walking along the front reveals
the fundamental budget trade-off: better signal recovery costs more.

```python
--8<-- "examples/monitoring_study.py:results"
```

## What to try next

- Use `build_grid(factors, method="lhs", n_samples=60)` for a Latin
  hypercube over continuous factor bounds.
- Wrap the sweep in a multi-phase `Study` — screen first with a coarse
  grid, then refine the promising region.
- Add an `Annotation` for power consumption or maintenance cost from
  an external costing sheet.
- Swap the brick-wall filter for a Butterworth via `scipy.signal` to
  study filter roll-off effects.
