# Choosing ensemble weights

Choose weights according to the quantity you intend to optimize:

| Function | Meaning | Dependency |
| --- | --- | --- |
| `stack_scores` | Optimize the weighted mean of supplied per-model scores; split exact best-mean ties uniformly | NumPy core |
| `stack_proportional` | Apply a direction-aware softmax to mean scores, controlled by a temperature in score units | NumPy core |
| `stack_bayesian` | Delegate predictive-density stacking or pseudo-BMA to `arviz.compare` | `stacking` extra; model log-likelihoods required |
| `ensemble_predict` | Compute a weighted arithmetic average of supplied prediction arrays | NumPy core |

## Linear scores and changes of units

For a score matrix with one row per model and one column per test point,
`stack_scores` minimizes the weighted mean input score by default. A linear
objective over simplex weights has an optimum on the best-mean model. No
iterative optimizer is needed. With `maximize=True`, the largest mean is best.

```python
import numpy as np
from trade_study import stack_scores

scores = np.array([[1.0, 1.0], [2.0, 2.0]])
assert stack_scores(scores).tolist() == [1.0, 0.0]
assert stack_scores(1e6 * scores).tolist() == [1.0, 0.0]
assert stack_scores(1e6 * scores, maximize=True).tolist() == [0.0, 1.0]
assert stack_scores(np.array([[1.0], [1.0], [2.0]])).tolist() == [0.5, 0.5, 0.0]
```

| Supplied means | Minimize weights | Maximize weights |
| --- | --- | --- |
| `[1, 2]` | `[1, 0]` | `[0, 1]` |
| `[1e6, 2e6]` | `[1, 0]` | `[0, 1]` |
| `[1, 1, 2]` | `[0.5, 0.5, 0]` | `[0, 0, 1]` |

The tie policy uses exactly equal computed means, subject to floating
precision. Near ties do not receive a tolerance-based split. To retain smooth
weight across near ties, use `stack_proportional`; scale its `temperature`
with the scores when changing units.

Scores must form a finite, non-empty two-dimensional matrix. Missing values,
infinities and empty rows/columns raise `ValueError`. An unrepresentable mean
raises `FloatingPointError`; it does not silently become uniform weights.

Optimizing supplied model scores does not optimize the loss of an averaged
prediction or the score of a mixture predictive distribution. Compute those
quantities separately, using independent evaluation data after selection.

## Reusable and read-only weights

`ensemble_predict` normalizes a private copy. Caller-owned arrays, views and
their backing storage retain their original values and writeability.

```python
from trade_study import ensemble_predict

predictions = [np.array([0.0, 2.0]), np.array([2.0, 6.0])]
weights = np.array([1.0, 3.0])
weights.setflags(write=False)
assert ensemble_predict(predictions, weights).tolist() == [1.5, 5.0]
assert weights.tolist() == [1.0, 3.0]
```

These examples' known optima, tie policy and read-only behavior are covered by
public API doctests and regression tests in the required test suite.
