# Shared-Map Objective Generalization

## Objective

Determine whether the observed cross-task alignment failures arise from the shared-map fitting
objective or persist after explicitly preserving probe-relevant information. The study tests map
generalization; it does not interpret residual failure as intrinsic concept mismatch.

## Design

Run the fixed protocol on the independently trained OLMo 1B pair and the SmolLM-1.7B pair,
separately. Use both directions, data seeds 42 and 137, the 75% residual depth, linear probes as
primary, and degree-2 and MLP probes as secondary. SST-2 and WildGuard are fitting tasks. AG News
and MNLI are held out from map fitting and selection.

For every task, reconstruct the pinned 12,000/2,000 probe splits. Reserve a fresh balanced
2,000-row evaluation set and disjoint 12,000/2,000 calibration/calibration-validation splits from
the unused training pool. These partitions must exclude the union of both seeds' probe splits and
the official test prompts. Abort before inference if any split or lineage contract fails.

## Conditions

Fit one map per model direction and seed using equal calibration rows from SST-2 and WildGuard:

1. fixed uniform affine ridge;
2. reconstruction-selected affine ridge over the prespecified regularization and variance-power
   grid;
3. probe-score-selected affine ridge over the identical candidate grid;
4. a shared probe-bank affine map built around the selected reconstruction map.

The candidate regularization grid is $10^{-6}$ through $10^0$ by powers of ten. Variance powers
are $0$, $0.25$, $0.5$, $0.75$, and $1$. Selection minimizes the worst fitting-task validation
error. Probe-score selection uses only frozen source-probe scores on paired calibration examples;
evaluation labels are unavailable to fitting and selection.

Matched quotient/full-ridge score equivalence and the ability of rank-one corrections to emulate
a linear-probe update are algebraic controls. Probe-bank success is therefore treated as monitor
preservation on the fitted bank, not evidence for a unique global coordinate map.

## Evaluation and decisions

Evaluate every condition on both fitting tasks first. Preserve the existing compatibility rule:
at least 75% median AUROC-gap recovery and same-task-improvement retention, at least 3/4
substantial recoveries, and no substantial shuffled-pair recovery. Only qualifying conditions are
evaluated on AG News and MNLI, without refitting.

The primary fitting outcome is worst-task median linear recovery. The primary generalization
outcome is held-out retention of same-task improvement. Retain AUROC, AUPRC, accuracy, balanced
accuracy, precision, recall, F1, calibration, confusion counts, low-FPR TPR, source-threshold
achieved FPR, thresholds, bootstrap intervals, diagnostics, and row-level predictions.

If stronger reconstruction alone succeeds, the earlier trade-off was fitting-procedure dependent.
If a fitting-task-compatible map degrades on held-out tasks, the result supports limited
task-general map recovery under the tested class. Task-specific low-rank adaptation is a separate,
subsequent study and will be attempted only after these zero-shot results are frozen.
