# OLMo 1 Pooled-Map Compatibility

[Hugging Face](https://huggingface.co/buckets/baimamboukar/coordinate-concept-mismatch/tree/studies) | [Weights & Biases](https://wandb.ai/JinesisLab/coordinate-concept-mismatch)

## Objective

Test whether the affine maps learned between the Ai2 and AMD OLMo 1B checkpoints represent a task-independent coordinate relation. Same-task maps previously recovered most frozen-probe transfer loss on both SST-2 and WildGuardMix, while maps fitted on one task transferred weakly to the other. This experiment distinguishes incompatible task-conditional maps from insufficient activation coverage or fit-sample budget.

## Formal comparison

Let $A_P$ be a ridge-regularized affine map fitted on the balanced pool $P=S\cup W$ of source-target activation pairs. For evaluation task $T\in\{S,W\}$, primary recovery is

$$
R_T(A_P)=\frac{\operatorname{AUROC}_T(A_P)-\operatorname{AUROC}^{\mathrm{frozen}}_T}{\operatorname{AUROC}^{\mathrm{oracle}}_T-\operatorname{AUROC}^{\mathrm{frozen}}_T}.
$$

Improvement retention compares the pooled map with the corresponding same-task affine map:

$$
I_T(A_P)=\frac{\operatorname{AUROC}_T(A_P)-\operatorname{AUROC}^{\mathrm{frozen}}_T}{\operatorname{AUROC}_T(A_T)-\operatorname{AUROC}^{\mathrm{frozen}}_T}.
$$

## Protocol

Both directions, seeds 42 and 137, and 75% residual depth are fixed. Linear probes and affine ridge are primary; degree-2 CP and MLP probes, orthogonal Procrustes, and shuffled affine pairing are secondary or negative controls. Models, tokenization, protected splits, thresholds, and metric definitions are unchanged from the completed task replications. Evaluation uses only held-out test rows. All primary and secondary metrics, confusion counts, low-FPR operating points, calibration, diagnostics, and row-level predictions are retained.

Two balanced fits are compared:

- **Equal budget:** 6,000 SST-2 plus 6,000 WildGuardMix rows, matching each earlier single-task map's total fit budget.
- **Full budget:** all 12,000 training rows from each task, testing whether additional observations resolve estimation error.

## Decision rule

A pooled map is compatible if, on **each** task, median primary recovery and same-task improvement retention are at least 75%, at least 3/4 directional-seed comparisons show substantial recovery, and shuffled controls show none. At least 90% median recovery and retention with 4/4 substantial comparisons is strong compatibility. A ten-point gain from equal to full budget indicates material sample-budget sensitivity. If full pooling remains below compatibility on either task while both established same-task maps pass it, the result supports task-conditional affine recoverability under this protocol, not the nonexistence of every possible global transformation.
