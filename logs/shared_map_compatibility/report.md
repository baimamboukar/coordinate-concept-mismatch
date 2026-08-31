# August 31, 2026 | Shared-Map Compatibility

[Hugging Face artifacts](https://huggingface.co/buckets/baimamboukar/coordinate-concept-mismatch/tree/studies/shared-map-compatibility/smollm-shared-map-compatibility) | [W&B runs — private project](https://wandb.ai/JinesisLab/coordinate-concept-mismatch/groups/shared_map_compatibility) | [Protocol](plan.md)

## Objective and design

We tested whether task-scale imbalance or ridge regularization explains the SmolLM shared-map compatibility failure. We reused the SmolLM-1.7B/SmolLM2 intermediate-checkpoint activations and frozen probes, both directions, data seeds 42/137, and 75% residual depth. Each affine map used the same pooled SST-2/WildGuard training activations. Uniform versus inverse-source-variance weighting was crossed with fixed versus validation-selected regularization. Selection used unlabelled fitting-task validation activations; no probe was retrained.

## Results

No condition passed compatibility on both included tasks. Consequently, AG News/MNLI evaluation was skipped, as prescribed.

Entries report median linear AUROC-gap recovery / retention of same-task improvement across the four direction–seed comparisons.

| Fitting condition | SST-2 | WildGuard |
|---|---:|---:|
| Uniform, fixed ridge | 66.9% / 67.4% | 90.2% / 99.8% |
| Scale-balanced, fixed ridge | 96.5% / 97.3% | 63.6% / 69.0% |
| Uniform, selected ridge | 67.4% / 68.0% | 90.7% / 100.4% |
| Scale-balanced, selected ridge — primary | 96.6% / 97.2% | 64.6% / 70.4% |

The primary intervention reduced worst-task median recovery from 66.9% to 64.6%. SST-2 passed with 4/4 substantial recoveries; WildGuard failed with 2/4. No shuffled control achieved substantial recovery in any condition (0/32).

WildGuard source-feature variance was approximately 10.5–281 times SST-2 variance, depending on model and seed. Balancing this disparity improved sentiment transfer but impaired safety transfer. Regularization selection made comparatively small changes; all four primary fits selected the largest prespecified coefficient, so the grid does not exhaust regularization alternatives.

## Secondary evidence and interpretation

Primary-condition median AUROC was 0.944/0.841 on SST-2/WildGuard; test-ROC TPR at 1% FPR was 52.0%/23.9%. At fixed source-calibrated 1%-FPR thresholds, achieved target FPR was 0.47%/4.92%. Degree-2 and MLP probes repeated the trade-off: recovery was 98.6%/97.7% on SST-2 versus 64.8%/65.5% on WildGuard. Paired-bootstrap intervals, remaining secondary metrics, confusion counts, and row-level predictions are in the artifacts.

These results establish fitting-objective dependence within the tested procedures, not intrinsic concept mismatch or the impossibility of a shared affine map. This remains exploratory: test sets were previously inspected, and alignment fitting overlaps probe training. Next, separate calibration from probe training and reserve an untouched evaluation set before stronger claims about task-general alignment. Preserve these results rather than tuning against the same tests.

All outputs were verified on Hugging Face, all eight W&B runs finished, and worker `ccm-842917` was destroyed. Heavy artifacts never passed through the Mac.
