# Paired comparisons

Use common random numbers in your simulator: replicate index `rep` must identify
the same random realization across configurations. Matching replicate labels
alone does not establish this scientific assumption. Keep raw replicated grid
results for these comparisons; aggregated means cannot recover paired draws.

`paired_difference()` estimates the mean difference A minus B and a bootstrap or
paired-t interval. `paired_rank()` compares designs with a reference and applies
Bonferroni adjustment by default. These comparisons address one observable;
they do not replace multi-objective Pareto analysis.

::: trade_study.PairedDifference

::: trade_study.paired_difference

::: trade_study.paired_rank
