# August 31, 2026 | SmolLM Held-Out Map Replication

[Hugging Face artifacts](https://huggingface.co/buckets/baimamboukar/coordinate-concept-mismatch/tree/studies/heldout-map-replication/smollm-heldout-map-replication) | [W&B runs — private project](https://wandb.ai/JinesisLab/coordinate-concept-mismatch/groups/heldout_map_replication) | [Protocol](plan.md)

## Outcome

This was a partial replication, not a clean confirmation of the OLMo result. Frozen-transfer failure and strong same-task linear recovery replicated, but the pooled map failed an included-task compatibility control. The preregistered broad-replication criterion was therefore not met.

We compared SmolLM-1.7B with the architecture-matched SmolLM2 checkpoint at step 5,125,000, using a shared tokenizer. Published recipes indicate independent initialization, but pretraining data and budgets differ: this is not a pure seed intervention. The fixed protocol retained four tasks, seeds 42/137, both directions, 75% residual depth, linear primary probes, nonlinear secondary probes, and 2,000-resample paired bootstrap intervals.

## Primary evidence

All four tasks met the frozen-transfer failure rule in 4/4 linear comparisons. Same-task affine recovery reached 99.2% on SST-2 and 90.1% on WildGuard, alongside the held-out-task controls below. Each had 4/4 substantial recoveries and no substantial shuffled controls.

Values are median per-comparison linear recovery across two directions and two data seeds.

| Evaluation task | Same-task fit | SST-2 fit | WildGuard fit | Equal pool | Full pool |
|---|---:|---:|---:|---:|---:|
| AG News | 96.6% | 54.6% | 57.6% | 65.4% | 72.5% |
| MNLI | 97.7% | 30.7% | 48.7% | 40.0% | 53.4% |

Neither primary pooled condition passed the held-out criterion. AG News full-pool retention was 74.7%, narrowly below 75%, despite 4/4 substantial recoveries; this is borderline. MNLI retained 54.6%, with 2/4 substantial recoveries. Held-out shuffled controls passed 0/32 comparisons.

Doubling fit data increased recovery by 7.1 points on AG News and 13.4 on MNLI; only MNLI met the prespecified ten-point materiality threshold.

Crucially, equal/full pooling recovered only 64.4%/66.9% on included SST-2, failing compatibility. WildGuard recovered 88.5%/90.1% and passed. Unlike OLMo, held-out degradation cannot be isolated from shared-fit interference here.

## Secondary evidence and interpretation

Orthogonal full-pool maps recovered 67.5%/65.3% on AG News/MNLI, showing estimator dependence. Their high AG News retention is relative to a weaker same-task orthogonal baseline. Degree-2 same-task affine recovery was substantial in only 2/4 AG News comparisons, limiting interpretation of its held-out losses.

Linear full-pool TPR at 1% FPR was 28.1%/6.8%, versus within-model 75.6%/56.2%. Precision, recall, balanced accuracy, calibration, achieved source-threshold FPR, confusion counts, and row-level predictions remain available in the artifacts.

Next, diagnose included-task interference through activation-scale/task-weighting, regularization, and fit-budget/coverage controls using the existing caches. Restore compatibility before interpreting held-out degradation or introducing task-specific low-rank corrections. These results do not establish intrinsic concept mismatch.

All outputs were verified on Hugging Face, all 20 W&B runs finished, and the worker was destroyed. Activations never passed through the Mac.
