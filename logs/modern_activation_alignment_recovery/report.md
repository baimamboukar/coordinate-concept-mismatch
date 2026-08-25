# August 25th, 2026 | Modern Activation-Alignment Recovery

[Hugging Face artifacts](https://huggingface.co/buckets/baimamboukar/coordinate-concept-mismatch/tree/experiments/modern_activation_alignment_recovery) | [Weights & Biases](https://wandb.ai/JinesisLab/coordinate-concept-mismatch/runs/bz38ys3e) | [Experiment plan](plan.md)

## Summary

This experiment tests how much of the natural frozen-probe transfer gap can be recovered by alignment fitted on paired, unlabeled activations. The primary permutation-plus-positive-diagonal map passed the prespecified substantial-recovery rule in all four 75%-depth linear Llama–Qwen comparisons: both directions under seeds 42 and 137. It improved AUROC by a median 0.317, recovered 73.0% of the target-oracle gap, and left a median residual gap of 0.115. Recovery ranged from 66.5% to 77.6%, with every paired 95% bootstrap interval for improvement above zero.

Strict permutation recovered 79.1% of the gap. Orthogonal, affine, and quotient maps recovered 93.6–95.1%, while shuffled-pair affine alignment recovered only 4.0% and passed the recovery rule in 0 of 4 comparisons. The degree-2 and MLP probes also passed in 4 of 4 comparisons, with median restricted recovery of 82.8% and 84.1%.

The Llama–Nemotron lineage control was non-degraded in all four comparisons; AUROC changes ranged from -0.001 to 0.009, and source-threshold FPR moved closer to the nominal 1% and 5% targets in every direction-seed-threshold check. Exploratory Qwen–Nemotron recovery passed in 2 of 4 comparisons and remained strongly direction-dependent. All primary and secondary metrics, confusion counts, thresholds, diagnostics, and row-level predictions passed validation.

## Interpretation

Restricted activation alignment explains a substantial part of the independent Llama–Qwen transfer gap, while the failed shuffled control shows that recovery depends on paired examples. Flexible-map recovery is a linear-recoverability upper bound, not evidence of an exact parameter symmetry. The remaining restricted gap and the Qwen–Nemotron asymmetry prevent a coordinate-only conclusion. The next step is to add Mistral under the same primary-depth protocol before expanding the analysis across layers.
