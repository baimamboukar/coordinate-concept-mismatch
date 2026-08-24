# August 23rd, 2026 | Frozen Probe Transfer Baseline

[Pythia pilot artifacts](https://huggingface.co/buckets/baimamboukar/coordinate-concept-mismatch/tree/experiments/pythia_seed_probe_transfer) | [Weights & Biases](https://wandb.ai/JinesisLab/coordinate-concept-mismatch/runs/hd5rdpqo) | [Experiment plan](plan.md)

## Summary

Experiment 1 tests whether a probe trained on one model transfers unchanged to another. We first ran the full protocol on two independently trained Pythia-410M checkpoints as a low-cost pilot. Target-trained probes reached AUROC 0.834–0.877, while frozen-transfer AUROC fell to 0.403–0.572. All 12 primary comparisons met the prespecified failure rule. This is pilot evidence for the baseline, not a separate experiment or a broad cross-family result.

## Pythia pilot results

Each cell reports target-oracle AUROC → frozen-transfer AUROC, followed by the gap in parentheses.

| Data seed | Direction | Linear | Degree-2 | MLP |
| ---: | --- | ---: | ---: | ---: |
| 42 | seed 1234 → seed 1 | 0.864 → 0.442 (0.422) | 0.865 → 0.540 (0.325) | 0.877 → 0.499 (0.378) |
| 42 | seed 1 → seed 1234 | 0.859 → 0.471 (0.388) | 0.869 → 0.451 (0.418) | 0.875 → 0.504 (0.372) |
| 137 | seed 1234 → seed 1 | 0.855 → 0.406 (0.450) | 0.834 → 0.572 (0.262) | 0.862 → 0.403 (0.459) |
| 137 | seed 1 → seed 1234 | 0.856 → 0.531 (0.325) | 0.836 → 0.483 (0.353) | 0.865 → 0.503 (0.363) |

Every paired 95% bootstrap interval excluded zero. At 1% FPR, median TPR fell from 13.9–22.4% for target-trained probes to 1.0–1.3% after frozen transfer. Full metrics, predictions, activations, probes, and thresholds are in the linked Hugging Face folder.

## Interpretation

The pilot establishes that raw frozen-probe transfer can fail even across same-architecture training runs, and nonlinear probe capacity does not remove the gap. It does not establish broad failure across modern model families or explain whether the mismatch is coordinate-based or conceptual.

## Progression

The residual-permutation control can now test whether coordinate mismatch alone is sufficient. Llama, Mistral, Qwen, and Nemotron remain the pending higher-cost phase of this same baseline experiment and will extend—not replace—the Pythia result.
