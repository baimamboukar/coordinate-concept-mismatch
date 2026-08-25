# August 25th, 2026 | Frozen Probe Transfer Baseline

[Plan](plan.md) | [Cross-family artifacts](https://huggingface.co/buckets/baimamboukar/coordinate-concept-mismatch/tree/experiments/frozen_probe_transfer_baseline/cross_family_extension) | [Cross-family W&B run](https://wandb.ai/JinesisLab/coordinate-concept-mismatch/runs/fpydnkqn) | [Earlier modern artifacts](https://huggingface.co/buckets/baimamboukar/coordinate-concept-mismatch/tree/experiments/frozen_probe_transfer_baseline/modern_phase) | [Pythia pilot artifacts](https://huggingface.co/buckets/baimamboukar/coordinate-concept-mismatch/tree/experiments/pythia_seed_probe_transfer)

## Summary

Frozen linear probes failed to transfer in all 20 prespecified 75%-depth direction-seed comparisons among Llama, Qwen, Mistral, and Granite. Target-trained probes reached AUROC 0.915–0.934, while unchanged cross-model probes reached 0.396–0.684. The transfer gap ranged from 0.250 to 0.525, with a median of 0.411; every paired 95% bootstrap interval excluded zero and every comparison met the failure rule. The experiment therefore satisfies its broad cross-family criterion.

| Direction | AUROC gap, seed 42 / 137 |
| --- | ---: |
| Llama $\rightarrow$ Mistral | 0.320 / 0.363 |
| Mistral $\rightarrow$ Llama | 0.428 / 0.432 |
| Qwen $\rightarrow$ Mistral | 0.413 / 0.425 |
| Mistral $\rightarrow$ Qwen | 0.394 / 0.411 |
| Llama $\rightarrow$ Granite | 0.525 / 0.382 |
| Granite $\rightarrow$ Llama | 0.428 / 0.389 |
| Qwen $\rightarrow$ Granite | 0.278 / 0.340 |
| Granite $\rightarrow$ Qwen | 0.301 / 0.250 |
| Mistral $\rightarrow$ Granite | 0.411 / 0.507 |
| Granite $\rightarrow$ Mistral | 0.476 / 0.468 |

At a target-specific 1% FPR, frozen-transfer TPR was 0–4.4%, compared with 31.2–48.1% for target-trained probes. Source-selected thresholds were also unstable: some transfers detected no positives, while others produced target FPR as high as 100%. All secondary metrics, thresholds, confusion counts, and row-level outputs passed validation and are retained in the public artifact directory.

The earlier phases give two useful controls. All 24 Llama–Qwen comparisons failed, whereas the Llama–Nemotron lineage control failed in 0 of 24; the Pythia seed pilot failed in all 12 primary comparisons. Degree-2 and MLP probes did not eliminate the Pythia gap.

## Interpretation

Probe transfer failure is broad across the four independent model families tested, not an isolated Llama–Qwen result. The preserved Llama–Nemotron transfer shows that failure is not automatic whenever checkpoints differ. This baseline establishes the phenomenon but does not identify its cause; the next experiment measures how much of each gap can be recovered by label-free activation alignment.
