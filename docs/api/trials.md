# Event-Driven Trials

Exact, simulation-free design calculations for two-arm comparative trials
whose primary analysis counts events, such as vaccine-efficacy or
prophylaxis trials with a success criterion on `efficacy = 1 - rate ratio`.

Given `D` total events and treatment-to-control person-time ratio `rho`,
treatment-arm events are `Binomial(D, theta)` with
`theta = rho (1 - E) / (rho (1 - E) + 1)`. Conditioning on `D` removes the
unknown baseline rate, so the efficacy interval, power and group-sequential
boundaries are exact binomial calculations.

```python
from trade_study import (
    design_group_sequential,
    events_required,
    operating_characteristics,
)

events_required(0.7, margin=0.3)  # 78 events for 90% power
design = design_group_sequential([39, 78], margin=0.3, spending="obf")
operating_characteristics(design, 0.7).power
```

**Assumptions the caller establishes.** Proportional event rates over
follow-up, a known person-time ratio, and independent events across
participants. Clustering (for example by village or household), outcome
misclassification and time-varying effects inflate variance or bias the
estimate. Assess them by simulation with [`run_grid`](runner.md) and
[`power_curve`](power.md).

**Discreteness.** Exact power is saw-toothed in the number of events.
`events_required` returns the smallest target beyond which power never falls
below the goal. Group-sequential boundaries spend at most the Lan-DeMets
alpha available at each look, so the alpha actually spent (`alpha_spent`) can
be below nominal; unspent alpha carries forward to later looks. Futility is
non-binding: boundaries ignore it, so type I error is controlled whether or
not it is applied.

**Accrual.** `expected_events_by_period` integrates expected events under
staggered uniform enrolment, exponential dropout and a cosine seasonal
hazard. `periods_to_events` reads off when an event target is expected.

All functions require SciPy (`trade-study[stats]`).

::: trade_study.efficacy_share

::: trade_study.efficacy_from_share

::: trade_study.EfficacyInterval

::: trade_study.efficacy_interval

::: trade_study.conditional_power

::: trade_study.events_required

::: trade_study.unconditional_power

::: trade_study.alpha_spending

::: trade_study.GroupSequentialDesign

::: trade_study.design_group_sequential

::: trade_study.OperatingCharacteristics

::: trade_study.operating_characteristics

::: trade_study.max_events_for_power

::: trade_study.AccrualPlan

::: trade_study.expected_events_by_period

::: trade_study.periods_to_events
