# August 23rd, 2026 | Pythia Seed Probe Transfer

[Hugging Face artifacts](https://huggingface.co/buckets/baimamboukar/coordinate-concept-mismatch/tree/experiments/pythia_seed_probe_transfer) | [Weights & Biases](https://wandb.ai/JinesisLab/coordinate-concept-mismatch/runs/hd5rdpqo) | [Experiment plan](plan.md)

## Summary

We tested whether safety probes trained on one Pythia-410M training run transfer unchanged to another independently trained Pythia-410M checkpoint. Using the same WildGuardMix task, we trained linear, degree-2 CP, and one-hidden-layer MLP probes with two data seeds and evaluated both transfer directions. Target-trained probes were strong, with AUROC between 0.834 and 0.877, but frozen-transfer AUROC fell to 0.403–0.572. All 12 primary transfers, and all 24 transfers evaluated across probe families and depths, met the failure rule defined in the plan.

## Results

Each cell reports target-oracle AUROC → frozen-transfer AUROC, followed by the gap in parentheses.

| Data seed | Direction | Linear | Degree-2 | MLP |
| ---: | --- | ---: | ---: | ---: |
| 42 | seed 1234 → seed 1 | 0.864 → 0.442 (0.422) | 0.865 → 0.540 (0.325) | 0.877 → 0.499 (0.378) |
| 42 | seed 1 → seed 1234 | 0.859 → 0.471 (0.388) | 0.869 → 0.451 (0.418) | 0.875 → 0.504 (0.372) |
| 137 | seed 1234 → seed 1 | 0.855 → 0.406 (0.450) | 0.834 → 0.572 (0.262) | 0.862 → 0.403 (0.459) |
| 137 | seed 1 → seed 1234 | 0.856 → 0.531 (0.325) | 0.836 → 0.483 (0.353) | 0.865 → 0.503 (0.363) |

Every paired 95% bootstrap interval excluded zero. At 1% FPR, median TPR fell from 13.9–22.4% for target-trained probes to 1.0–1.3% after frozen transfer. Full metrics, thresholds, predictions, activations, and trained probes are available in the Hugging Face folder above.

## Interpretation

Frozen probe transfer clearly fails for this controlled same-architecture checkpoint pair, and adding nonlinear probe capacity does not solve the problem. This establishes the baseline required by the plan. It does not yet show whether the failure comes from coordinate mismatch or different learned concepts, and it should not be generalized to modern model families.

## Next

Apply an exactly function-preserving symmetry transformation to one checkpoint, verify identical model outputs, and compare raw probe transfer with analytically transported probes. We can then test symmetry-aware alignment between the two independently trained checkpoints.
