# August 28, 2026 | OLMo 1 Pooled-Map Compatibility

[Plan](plan.md) | [Hugging Face artifacts](https://huggingface.co/buckets/baimamboukar/coordinate-concept-mismatch/tree/studies/olmo1-pooled-map-compatibility) | W&B: [equal SST-2](https://wandb.ai/JinesisLab/coordinate-concept-mismatch/runs/nbn07gdz), [full SST-2](https://wandb.ai/JinesisLab/coordinate-concept-mismatch/runs/tkd13fc3), [equal WildGuard](https://wandb.ai/JinesisLab/coordinate-concept-mismatch/runs/74ukxooh), [full WildGuard](https://wandb.ai/JinesisLab/coordinate-concept-mismatch/runs/w17niphj)

The preceding experiment found that affine maps fitted on SST-2 transferred weakly to WildGuardMix and vice versa, despite near-complete same-task recovery. We tested whether this reflected incompatible task-conditional maps or insufficient concept coverage.

One affine map was fitted on a balanced union of paired Ai2 and AMD OLMo 1B activations, then evaluated separately on protected SST-2 and WildGuardMix test splits. The equal-budget condition used 6,000 training rows per task; the full condition used 12,000 per task. Both directions, seeds 42 and 137, and 75% residual depth were fixed. Linear probes were primary; degree-2 and MLP probes, orthogonal alignment, and shuffled pairing were retained as controls.

| Evaluation | Fit budget | Median recovery | Median same-task retention | Substantial | Shuffled |
|---|---:|---:|---:|---:|---:|
| SST-2 | 12,000 | 83.0% | 83.3% | 3/4 | 0/4 |
| SST-2 | 24,000 | 85.7% | 85.9% | 3/4 | 0/4 |
| WildGuardMix | 12,000 | 94.1% | 99.0% | 4/4 | 0/4 |
| WildGuardMix | 24,000 | 94.8% | 99.7% | 4/4 | 0/4 |

All four conditions passed their exact output contracts and the preregistered compatibility rule. Doubling the fit budget improved median recovery by only 2.6 points on SST-2 and 0.8 points on WildGuardMix, below the ten-point sample-sensitivity threshold. Full-budget degree-2 recovery was 89.2% and 91.7%; MLP recovery was 89.7% and 94.4%.

Thus, the earlier failure of a map learned on one task to serve another did **not** establish incompatible model coordinates. A balanced multi-concept fit produced one map that served both tasks, implying that activation-distribution coverage was the main missing ingredient under this protocol. The conclusion is deliberately probe-visible: SST-2 affine reconstruction error remained high, and one directional-seed comparison recovered only 40.4%. Monitor recovery therefore does not establish full representational equivalence.

The next experiment should test this pooled map without refitting on a separately preregistered, linearly decodable task panel. BoolQ remains non-qualifying and will not be substituted within its completed experiment.
