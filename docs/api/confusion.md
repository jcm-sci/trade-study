# Confusion Tables

A replicated recovery or discrimination study generates independent cohorts
from each candidate structure and reads every cohort with a prespecified rule,
for example "favour A", "favour B", "indistinguishable" or "failed".
`confusion_table` tabulates the verdicts by generating structure and gives a
fixed-sample [proportion interval](proportions.md) for every cell, so the
probability of the right verdict is reported beside the wrong-structure and
no-verdict rates.

```python
table = confusion_table(truth, verdict, rows=["A", "B"],
                        columns=["A", "B", "none", "failed"],
                        adjustment="bonferroni")
table.cell("A", "A")   # probability of favouring A when A generated the cohort
table.row_total("A")   # cohorts generated from A, failed trials included
```

Every cohort is in a declared row and column: undeclared labels raise, a
declared row without a cohort raises, and a failed trial is a column of its
own, never a dropped cohort. The intervals assume independent cohorts and a
prespecified number per row; the caller establishes both. Without
`adjustment` each interval is marginal; `adjustment="bonferroni"` uses level
`1 - (1 - confidence) / len(columns)` so that the statements about the columns of
one row hold simultaneously at approximately the nominal level (Wilson coverage is
approximate, and different rows are separate statements that are not adjusted).
Labels that compare equal (`1`, `True`, `1.0`) are one label. `power_curve` reads one 0/1 indicator per
design; the confusion table reports all verdicts of a design together.

::: trade_study.ConfusionTable

::: trade_study.confusion_table
