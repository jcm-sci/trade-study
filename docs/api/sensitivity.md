# Post-hoc sensitivity

`sensitivity_from_table()` fits a surrogate over completed experiments and
estimates Morris or Sobol sensitivity cheaply through that model. Inspect its
cross-validated accuracy before interpreting importance: this measures the
surrogate's estimated response, not a new set of simulator evaluations.
Use `sobol_indices()` when you need first-order and total-order indices together;
`ST - S1` reveals effects attributable to interactions.

See the complete [conditional sensitivity and discrete contrasts workflow](../guide/table-sensitivity.md)
for explicit fixed-stratum selection, refusal of varying omitted inputs and
explicit configuration/dataset grouped validation and its one-partition boundary.

::: trade_study.TableSensitivity

::: trade_study.sensitivity_from_table
