# August 31, 2026 | Disjoint Calibration

[Hugging Face artifacts](https://huggingface.co/buckets/baimamboukar/coordinate-concept-mismatch/tree/studies/disjoint-calibration/smollm-disjoint-calibration) | [W&B runs — private project](https://wandb.ai/JinesisLab/coordinate-concept-mismatch/groups/disjoint_calibration) | [Protocol](plan.md)

## Objective and design

We tested whether overlap between probe-training and alignment-fitting examples explains the SmolLM shared-map trade-off. We retained the pinned SmolLM-1.7B/SmolLM2 intermediate-checkpoint pair, 75% residual depth, both directions, and data seeds 42/137. Probes were retrained once on the original 12,000/2,000 training/validation splits, then frozen across conditions.

Per task, we reserved a balanced 2,000-example fresh holdout and sampled 12,000 calibration plus 2,000 calibration-validation examples per seed. These excluded both seeds' original probe splits and all official-test prompts. Historical row identities and split isolation passed verification. Overlapping versus disjoint fitting was crossed with uniform versus inverse-source-variance weighting, with the relative ridge coefficient fixed at 0.0001. Calibration-validation inputs supplied diagnostics only.

## Results

Both tasks qualified: frozen linear transfer failed in 4/4 direction–seed comparisons per task, while disjoint same-task alignment recovered 98.0% on SST-2 and 91.6% on WildGuard, with 4/4 substantial recoveries each.

No pooled condition passed compatibility on both tasks. Entries below are median linear AUROC-gap recovery across four direction–seed comparisons.

| Fitting condition | SST-2 | WildGuard |
|---|---:|---:|
| Overlapping, uniform | 70.7% | 93.5% |
| Disjoint, uniform | 72.3% | 93.0% |
| Overlapping, scale-balanced | 96.3% | 70.1% |
| Disjoint, scale-balanced — primary | 95.3% | 71.4% |

Primary-condition retention of same-task improvement was 97.6%/77.2%; substantial recovery occurred in 4/4 and 3/4 comparisons. WildGuard remained below the 75% median-recovery gate. No primary shuffled control achieved substantial recovery (0/40, including references).

Primary paired AUROC changes, disjoint minus overlapping, had medians −0.0041 on SST-2 and −0.0032 on WildGuard. Seven of eight unadjusted 95% bootstrap intervals (2,000 resamples) included zero; the remaining interval was negative. Uniform-weight changes were +0.0036/−0.0034, again with no positive interval excluding zero. Paired changes and normalized recovery are different summaries; their medians need not move together.

## Secondary evidence and interpretation

Primary-condition AUROC was 0.942/0.875. Test-ROC TPR at 1% FPR was 51.4%/32.1%; source-calibrated 1%-FPR thresholds instead produced target FPRs of 1.45%/6.0%. Degree-2 and MLP recovery was 98.0%/96.6% on SST-2 versus 65.6%/69.6% on WildGuard. All remaining metrics, confusion counts, thresholds, and predictions are retained.

Removing training/calibration overlap did not restore joint compatibility under these procedures. This does not establish intrinsic concept mismatch or exclude other shared fitting objectives. The fresh holdout came from unused training-pool records, so absolute scores are not directly comparable with earlier official-test results; semantic near-duplicates and pretraining contamination remain uncontrolled.

Next, compare a validation-selected shared map with small task-specific corrections under fixed rank/data budgets and a separately locked evaluation protocol. AG News/MNLI were outside this run.

The audit recomputed all classification metrics from saved predictions, verified direct Hugging Face publication and all twelve W&B runs, and confirmed worker `ccm-731406` was destroyed. Heavy artifacts never passed through the Mac.
