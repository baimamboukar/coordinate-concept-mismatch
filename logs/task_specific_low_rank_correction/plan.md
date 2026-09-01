# Task-Specific Low-Rank Correction

## Objective

Determine whether the failure of a task-independent cross-model map reflects a completely different coordinate system or a small task-conditioned deviation. The shared maps and frozen-transfer results from `shared_map_objective_generalization` are immutable inputs.

## Formal setup

For a frozen target-to-source affine map $$A_0(h)=hW_0+b_0$$, fit only a rank-constrained task correction on paired, unlabeled activations from held-out task $$t$$:

$$
A_t(h)=h(W_0+\Delta W_t)+b_0,\qquad \operatorname{rank}(\Delta W_t)\le r.
$$

The correction is a truncated ridge solution to the residual $$h_s-A_0(h_t)$$. The shared map and bias remain fixed. A shuffled-pair correction is the negative control.

## Protocol

- Model pairs: SmolLM-1.7B/SmolLM2-1.7B and AI2/AMD OLMo-1B.
- Shared-map fits: SST-2 and WildGuardMix; probe-selected is primary and reconstruction-selected is secondary.
- Held-out adaptation/evaluation tasks: AG News and MNLI.
- Probe: frozen linear probe at normalized depth 0.75.
- Data seeds: 42 and 137; both transfer directions are evaluated.
- Adaptation budgets: 64, 256, 1,024, and 4,096 paired calibration activations.
- Correction ranks: 1, 2, 4, 8, 16, and 32.
- Confirmatory endpoint: rank 8 with 256 pairs, fixed before test evaluation.
- Diagnosis uses the disjoint calibration-validation split; protected test labels never enter fitting or selection.

## Comparison and decision rule

The ordered comparison is frozen transfer, shared global map, shared map plus low-rank correction, full same-task affine reference, and shuffled controls. The confirmatory endpoint passes when median recovery is at least 0.50, median retention of same-task affine improvement is at least 0.75, at least three of four seed-direction comparisons show substantial recovery, and no shuffled comparison does.

Primary metrics are recovery fraction, improvement retention, and aligned AUROC improvement. Secondary outputs retain AUROC, AUPRC, accuracy, balanced accuracy, precision, recall, F1, calibration error, confusion counts, thresholds, TPR at 1% FPR, bootstrap intervals, diagnostics, and row-level predictions.

All compute runs from YAML through the generic pipeline. Inputs are materialized from public Hugging Face artifacts on the worker; outputs publish directly to Hugging Face and training telemetry to W&B.
