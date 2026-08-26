# August 25th, 2026 | Modern Activation-Alignment Recovery

[Plan](plan.md) | [Cross-family artifacts](https://huggingface.co/buckets/baimamboukar/coordinate-concept-mismatch/tree/studies/modern-activation-alignment-recovery/modern-models/cross-family) | [Cross-family W&B run](https://wandb.ai/JinesisLab/coordinate-concept-mismatch/runs/kf9jhj5a) | [Reference and lineage artifacts](https://huggingface.co/buckets/baimamboukar/coordinate-concept-mismatch/tree/studies/modern-activation-alignment-recovery/modern-models/reference-and-lineage) | [Earlier W&B run](https://wandb.ai/JinesisLab/coordinate-concept-mismatch/runs/bz38ys3e)

## Summary

The primary permutation-plus-positive-diagonal alignment recovered a median 61.8% of the frozen-probe AUROC gap across the 20 new direction-seed comparisons. Median AUROC improvement was 0.251 and the median residual oracle gap was 0.155. Sixteen comparisons met the substantial-recovery rule, so the prespecified all-20 broad restricted-coordinate claim was not satisfied.

The four exceptions were Qwen $\rightarrow$ Granite under seed 42 (49.98% recovery), Mistral $\rightarrow$ Granite under seed 42 (48.4%), and Granite $\rightarrow$ Qwen under both seeds. Granite $\rightarrow$ Qwen was the clear failure: alignment worsened AUROC under seed 42 and recovered only 5.5% under seed 137, whose improvement interval included zero. Every other direction-seed comparison passed.

| Alignment | Passed | Median gap recovered |
| --- | ---: | ---: |
| Permutation + positive diagonal | 16/20 | 61.8% |
| Permutation | 19/20 | 65.7% |
| Orthogonal Procrustes | 20/20 | 93.4% |
| Affine Ridge | 20/20 | 97.3% |
| Quotient Ridge | 20/20 | 97.3% |
| Shuffled-pair affine control | 0/20 | -3.5% |

At 1% FPR, median TPR rose from 1.0% before alignment to 18.2% after restricted alignment, compared with 39.0% for target-trained probes. The same restricted map passed 19/20 comparisons for both degree-2 and MLP probes, with median recovery of 70.4% and 65.6%, respectively. All primary and secondary metrics, thresholds, confusion counts, diagnostics, and row-level predictions passed validation and are retained in the public artifact directory.

The earlier Llama–Qwen phase passed its restricted-recovery rule in all four comparisons with median recovery of 73.0%. Together, the two phases show that label-free activation alignment usually repairs a substantial fraction of natural cross-model probe failure, but not universally.

## Interpretation

The failed shuffled control shows that recovery depends on prompt correspondence, while near-complete recovery by flexible linear maps establishes a strong linear-recoverability result. Neither result proves an exact parameter-space symmetry. The restricted map’s 16/20 success and the persistent Granite $\rightarrow$ Qwen failure argue against a coordinate-only explanation: coordinate mismatch is substantial, but model- and direction-specific residual differences remain.
