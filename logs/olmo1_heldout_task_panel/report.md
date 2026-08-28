# August 28, 2026 | OLMo 1 Held-Out Task Panel

[Hugging Face artifacts](https://huggingface.co/buckets/baimamboukar/coordinate-concept-mismatch/tree/studies/olmo1-heldout-map-generalization) | [Weights & Biases](https://wandb.ai/JinesisLab/coordinate-concept-mismatch)

## Objective

Test whether the affine activation map that served both SST-2 and WildGuardMix generalizes without refitting to preregistered, conceptually distinct tasks. AG News (World versus Business) and MNLI (entailment versus contradiction) used pinned data, Ai2 and AMD OLMo 1B checkpoints, seeds 42 and 137, both model directions, and 75% residual depth.

## Qualification

Both tasks satisfied the frozen-transfer gate in all 4/4 linear comparisons. Median target-oracle versus frozen AUROC was 0.981 versus 0.476 on AG News and 0.944 versus 0.533 on MNLI, producing gaps of 0.505 and 0.410. Same-task affine maps recovered 98.6% and 96.1% of the median gaps, respectively, with 4/4 substantial recoveries and 0/4 shuffled controls on each task. The held-out tests were therefore interpretable.

## Held-out results

Values are median linear recovery; parentheses give substantial directions out of four.

| Evaluation task | SST-2 fit | WildGuard fit | Equal pool | Full pool |
|---|---:|---:|---:|---:|
| AG News | 31.0% (1) | 54.8% (2) | 60.4% (3) | 57.1% (2) |
| MNLI | 26.2% (0) | 35.5% (0) | 32.3% (0) | 35.9% (0) |

No pooled condition met the fixed criterion. Equal-pool same-task improvement retention was 61.3% on AG News and 33.4% on MNLI, below 75%. Doubling the fit budget changed recovery by −3.2 and +3.6 points, below the ten-point materiality threshold. Primary shuffled controls passed 0/32 comparisons.

The result is not specific to linear AUROC. Under full pooling, degree-2/MLP median recovery was 50.0%/50.8% on AG News and 31.3%/42.4% on MNLI. Linear TPR at 1% FPR improved from 0.9% to 12.5% on AG News and from 1.1% to 3.3% on MNLI, but remained far below same-task oracle values of 73.1% and 43.6%.

## Finding and next test

For this independently trained OLMo pair, excellent included-task compatibility does not imply a task-general coordinate map. Flexible affine recovery is strongly concept-distribution-bound, so it cannot be interpreted as canonicalizing a single global representation. The next discriminating experiment should compare a shared map with a shared-plus-task-specific low-rank decomposition across additional fit tasks, followed by replication on a second independent model pair.
