# August 25th, 2026 | Frozen Probe Transfer Baseline

[Pythia pilot artifacts](https://huggingface.co/buckets/baimamboukar/coordinate-concept-mismatch/tree/experiments/pythia_seed_probe_transfer) | [Modern artifacts](https://huggingface.co/buckets/baimamboukar/coordinate-concept-mismatch/tree/experiments/frozen_probe_transfer_baseline/modern_phase) | [Pythia pilot on Weights & Biases](https://wandb.ai/JinesisLab/coordinate-concept-mismatch/runs/hd5rdpqo) | [Modern phase on Weights & Biases](https://wandb.ai/JinesisLab/coordinate-concept-mismatch/runs/dpsh1n4l) | [Experiment plan](plan.md)

## Summary

This experiment tests whether a probe trained on one model can be applied unchanged to another. The Pythia pilot first established failure across same-architecture seeds. The modern phase now establishes the prespecified pairwise result: all 24 Llama–Qwen comparisons failed, covering both directions, two data seeds, four depths, and the three probe families at the primary depth. The Llama–Nemotron lineage control failed in 0 of 24 comparisons, while the exploratory Qwen–Nemotron comparison failed in 24 of 24. This contrast is evidence that the baseline is detecting model-dependent representation mismatch rather than generic probe instability.

## Primary modern result

Values are reported as seed 42 / seed 137. The table shows the primary 75%-depth linear probe.

| Direction | Target oracle AUROC | Transfer AUROC | Gap [95% CI] |
| --- | ---: | ---: | ---: |
| Llama $\rightarrow$ Qwen | 0.933 / 0.934 | 0.599 / 0.503 | 0.334 [0.308, 0.361] / 0.432 [0.401, 0.461] |
| Qwen $\rightarrow$ Llama | 0.932 / 0.929 | 0.504 / 0.440 | 0.429 [0.399, 0.459] / 0.489 [0.459, 0.521] |
| Llama $\rightarrow$ Nemotron | 0.898 / 0.907 | 0.873 / 0.871 | 0.025 [0.012, 0.038] / 0.035 [0.022, 0.048] |
| Nemotron $\rightarrow$ Llama | 0.932 / 0.929 | 0.888 / 0.887 | 0.045 [0.035, 0.055] / 0.042 [0.032, 0.052] |

At a target-specific 1% FPR, Llama–Qwen transfer TPR was only 0.5–2.4%, compared with 37.4–48.1% for target-trained probes. Source thresholds selected for 1% FPR detected no positive examples after Llama–Qwen transfer. The lineage control preserved AUROC but not calibration: the Llama source threshold produced 23.1–25.7% FPR on Nemotron. All requested secondary metrics, confusion counts, thresholds, and row-level predictions passed validation.

## Pythia pilot

Across two Pythia-410M training seeds, target-trained probes reached AUROC 0.834–0.877 while frozen-transfer AUROC fell to 0.403–0.572. All 12 primary pilot comparisons met the failure rule, and adding degree-2 or MLP capacity did not remove the gap.

## Interpretation and next step

The baseline goal is achieved for the independent Llama–Qwen pair, with Llama–Nemotron providing a useful lineage-sensitive control. This is not yet a broad three-family result because Mistral remains pending. It also does not distinguish coordinate mismatch from concept mismatch. The completed Pythia symmetry study already showed that a coordinate change can cause and exactly repair probe failure; the next experiment therefore fits label-free alignments between the modern models and measures how much of each natural gap is recoverable, including source-threshold stability.
