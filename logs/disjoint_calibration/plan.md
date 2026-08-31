# Disjoint Calibration | Fresh-Holdout Control

## Objective

Determine whether overlap between probe-training and alignment-fitting examples affects the SmolLM shared-map result. The preceding weighting diagnostic did not restore compatibility on both SST-2 and WildGuard. This follow-up separates fitting-data overlap from task weighting without expanding the regularization search.

## Fixed design

Keep the pinned SmolLM-1.7B/SmolLM2 intermediate-checkpoint pair, tasks, prompts, token limit, 75% residual depth, both directions, and data seeds 42/137. Reconstruct the original 12,000-row probe-training and 2,000-row probe-validation splits exactly. Train probes once with the existing hyperparameter protocol, then freeze them across all alignment conditions. Linear probes are primary; degree-2 and MLP probes are secondary.

Exclude the union of both seeds' original training/validation prompts and all cleaned official-test prompts from the new calibration/evaluation pool, using normalized-text identity. Reserve one balanced 2,000-row fresh holdout per task with allocation seed 314. From the remaining pool, sample 12,000 calibration and 2,000 calibration-validation rows per seed, disjoint within seed. All calibration and evaluation rows must remain separate from every prior probe split. Abort before inference if fixed budgets or isolation checks fail.

The holdout is unused by the preceding split recipes, not a new corpus or a pretraining-contamination control. Because it comes from the training pool, absolute scores should not be compared directly with earlier official-test scores.

## Alignment and evaluation

Cross overlapping versus disjoint fitting with uniform versus inverse-source-variance weighting. Every pooled map uses 12,000 examples per task and the original fixed relative ridge coefficient, $10^{-4}$. Calibration-validation inputs provide diagnostics only. Fit same-task references on disjoint calibration data; repeat shuffled-pair controls. No alignment fitting or selection uses evaluation labels or scores.

The primary contrast is disjoint-minus-overlapping aligned AUROC under scale balancing, with 2,000-resample paired 95% bootstrap intervals. Report uniform-weight contrasts secondarily, alongside recovery, same-task retention, and worst-task recovery. Interpret compatibility only alongside fresh frozen-transfer and same-task qualification controls. Preserve the existing compatibility gate: each task needs 75% median recovery/retention, at least 3/4 substantial recoveries, and no substantial shuffled control.

Retain AUPRC, precision, recall, accuracy, balanced accuracy, F1, calibration, confusion counts, thresholds, achieved FPR, TPR at 1%/5% FPR, and row-level predictions. Evaluate every configured condition regardless of outcome; stop after this two-task control. AG News/MNLI are outside this run.

Publish necessary artifacts directly from the worker to Hugging Face, track training/fits in W&B, verify outputs, and destroy the six-digit-labeled worker.
