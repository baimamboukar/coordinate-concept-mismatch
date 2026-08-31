# Shared-Map Compatibility | Fitting Interference

## Objective

Determine whether task-scale imbalance or ridge regularization explains the SmolLM pooled-map compatibility failure. SST-2 previously recovered 99.2% with a same-task map but 66.9% with a full pooled map. This diagnostic follow-up tests fitting choices, not intrinsic concept mismatch; previously inspected test sets cannot provide a new confirmatory replication.

## Fixed materials

Reuse the pinned SmolLM-1.7B/SmolLM2 intermediate-checkpoint activations and frozen probes from the [replication](../heldout_map_replication/report.md). Keep both directions, data seeds 42/137, 75% residual depth, all source-selected thresholds, and the same 12,000 training and 2,000 validation rows per fitting task. No language-model inference or probe retraining is required. Alignment fitting reuses the original training split, which overlaps probe training; this is a controlled comparison of existing fits, not the proposal's stricter disjoint-calibration protocol.

## Fitting conditions

Fit one task-independent target-to-source affine map to the full SST-2/WildGuard pool. For task $k$, let $v_k$ be its mean feature variance in source training activations. Compare uniform sample weights with weights proportional to $1/v_k$, normalized to mean one. Weighted means determine a single global intercept; no task-specific normalization or routing is used at evaluation.

$$
\min_{A,b}\sum_i w_i\lVert h_s(x_i)-h_t(x_i)A-b\rVert_2^2+\lambda\lVert A\rVert_F^2,
\qquad
\lambda=\alpha\,\operatorname{tr}(H_{t,c}^{\top}WH_{t,c})/d.
$$

Cross weighting with fixed $\alpha=10^{-4}$ or validation selection from $\{10^{-6},10^{-5},10^{-4},10^{-3},10^{-2}\}$. Select the smallest worst-task validation MSE divided by training variance; exact ties favor stronger regularization. Selection receives only unlabelled SST-2/WildGuard validation activations. Report every candidate's per-task scale, fitting-loss share, and train/validation reconstruction error. Repeat the full selection procedure for within-task shuffled-pair controls, preserving task marginals.

## Evaluation and decision

Scale-balanced, validation-selected fitting is the primary intervention; the other three conditions separate weighting and regularization effects. Linear probes are primary; degree-2 and MLP probes remain secondary. Primary outcomes are AUROC-gap recovery and retention of same-task improvement for each included task, plus worst-task performance. Retain 2,000-resample paired bootstrap 95% intervals.

Compatibility requires each task to reach 75% median recovery and retention, at least three substantial comparisons out of four, and no substantial shuffled control. Preserve these thresholds. Evaluate each prespecified condition on AG News/MNLI only if it passes both included-task gates; otherwise report the trade-off and stop that condition.

Secondary metrics include AUPRC, precision, recall, accuracy, balanced accuracy, F1, calibration, confusion counts, thresholds, achieved FPR, TPR at 1%/5% FPR, and row-level predictions. Reuse same-task references unchanged. Heavy materials remain on the worker and Hugging Face; track fits in W&B, verify publication, then destroy the worker.
