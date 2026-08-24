# August 24th, 2026 | Pythia Checkpoint-Alignment Recovery

[Hugging Face artifacts](https://huggingface.co/buckets/baimamboukar/coordinate-concept-mismatch/tree/experiments/pythia_checkpoint_alignment_recovery) | [Weights & Biases](https://wandb.ai/JinesisLab/coordinate-concept-mismatch/runs/wvm2zs5x) | [Experiment plan](plan.md)

## Summary

Experiment 2 tests how much of the natural probe-transfer gap between independently trained Pythia-410M checkpoints can be recovered using alignment fitted on paired, unlabeled activations. The primary permutation-plus-positive-diagonal map improved block-18 AUROC by a median 0.222 and recovered 60.3% of the target-oracle gap across both transfer directions, two data seeds, and three probe families. Recovery ranged from 48.9% to 77.2%, and 11 of 12 comparisons met the prespecified substantial-recovery rule. The median residual AUROC gap was 0.146.

Strict permutation recovered 53.7% of the gap. The broader orthogonal and affine maps recovered 97.0% and 97.4% respectively, while quotient Ridge recovered 97.7% for linear probes. In contrast, affine alignment trained on shuffled prompt pairs recovered only 6.4% median and met the substantial-recovery rule in 0 of 12 comparisons. Full primary, secondary, low-FPR, confusion-matrix, calibration, diagnostic, and row-level results are available in the linked artifacts.

## Interpretation

A substantial part of the same-architecture Pythia transfer failure is compatible with a restricted coordinate mismatch. Near-complete recovery by flexible linear maps establishes broader linear recoverability, not a known parameter symmetry. The remaining gap under the restricted map is unexplained by this alignment class and is not, by itself, evidence of concept mismatch.
