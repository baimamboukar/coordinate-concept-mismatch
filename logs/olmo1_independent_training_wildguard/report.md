# August 28, 2026 | OLMo 1 Independent-Training Safety Replication

[Plan](plan.md) | [Baseline artifacts](https://huggingface.co/buckets/baimamboukar/coordinate-concept-mismatch/tree/studies/olmo1-independent-probe-transfer/olmo1-independent-training-wildguard/wildguard-baseline) | [Alignment artifacts](https://huggingface.co/buckets/baimamboukar/coordinate-concept-mismatch/tree/studies/olmo1-independent-alignment/olmo1-independent-training-wildguard/wildguard-alignment) | [Transfer W&B](https://wandb.ai/JinesisLab/coordinate-concept-mismatch/runs/zjezuasq) | [Alignment W&B](https://wandb.ai/JinesisLab/coordinate-concept-mismatch/runs/3nfn2he3)

## Summary

This experiment tested whether the independent-training result observed on SST-2 generalizes to WildGuardMix prompt harmfulness. The pinned Ai2 and AMD OLMo 1B checkpoints, shared tokenizer, tokenization settings, representation depths, probes, alignment methods, data seeds, and decision rules were unchanged.

The transfer-failure criterion passed. All four primary 75%-depth linear-probe comparisons failed: target-trained AUROC ranged from 0.882 to 0.891, transferred AUROC from 0.414 to 0.649, and the median AUROC gap was 0.461. The safety-task gap was therefore at least as pronounced as the 0.406 median gap previously observed on SST-2.

Restricted recovery again failed its prespecified rule. Permutation-plus-positive-diagonal alignment improved AUROC by a median 0.209 and recovered 44.3% of the gap, with substantial recovery in only 1/4 comparisons. Median residual gap remained 0.263, and one seed-direction comparison significantly worsened. Exact permutation recovered a median 53.1%, but passed in only 2/4 comparisons.

The broader alignment result replicated. Orthogonal Procrustes recovered a median 90.5% of the gap, while affine and quotient maps recovered approximately 94.9%; each was substantial in 4/4 comparisons. The shuffled-pair affine control was substantial in 0/4, despite 17.8% median apparent recovery. At a fixed 1% FPR, median TPR increased from 1.9% before alignment to 6.8% after restricted alignment and 17.8% after affine alignment, compared with 18.0% for the target-trained oracle. Calibration error similarly moved from 0.269 before alignment to 0.183 under the restricted map and 0.059 under affine alignment, versus 0.062 for the oracle.

Secondary results remained heterogeneous. Restricted median recovery was 52.1%, 35.8%, 44.3%, and 61.6% at 25%, 50%, 75%, and 100% depth, respectively. At the primary depth, degree-2, linear, and MLP probes recovered 38.1%, 44.3%, and 64.0%, with substantial recovery in 1/4, 1/4, and 3/4 comparisons.

The prespecified cross-task pattern therefore passes: independently trained, architecture-identical models exhibit large probe-transfer failures on both sentiment and harmfulness; sparse symmetry-like maps are inconsistent, while paired dense maps recover most of the gap and operational performance. This supports a broader coordinate or basis mismatch, but not an exact parameter-space symmetry. The next decisive test is cross-task map transport: fit alignment on one unlabeled prompt distribution and evaluate it on the other without refitting.
