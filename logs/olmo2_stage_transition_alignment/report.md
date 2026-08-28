# August 28, 2026 | OLMo 2 Stage-Transition Alignment

[Plan](plan.md) | [Hugging Face artifacts](https://huggingface.co/buckets/baimamboukar/coordinate-concept-mismatch/tree/studies/olmo2-stage-transition-alignment) | [W&B run](https://wandb.ai/JinesisLab/coordinate-concept-mismatch/runs/9nwz0ukp)

## Summary

This experiment tested whether the late-layer SST-2 probe-transfer failures between the shared OLMo 2 Stage-1 parent and three Stage-2 checkpoints could be repaired by unlabeled activation alignment. It was selected after the baseline exposed these failures, so it is a prespecified mechanistic follow-up rather than an independent replication. We retained every Stage-1↔Stage-2 direction, both data seeds, and the original failure gate.

The primary analysis passed. At 75% depth, all six eligible Stage-1→Stage-2 linear-probe failures satisfied the substantial-recovery criterion under permutation-plus-positive-diagonal alignment. Median recovery was 77.0%; mean AUROC improvement was 0.126, reducing the mean oracle gap from 0.165 to 0.040. Every improvement confidence interval excluded zero. Exact permutation recovered substantially in only 1/6 cases, while orthogonal, affine, and quotient maps did so in 6/6. Shuffled-pair alignment recovered none and reduced mean AUROC to approximately chance.

The effect was direction- and depth-specific. Reverse Stage-2→Stage-1 comparisons were not baseline failures, and alignment slightly reduced their mean AUROC. At 25% and 50% depth, the restricted map recovered none of the eligible failures; the final layer produced no eligible failures. As a secondary probe-family test, permutation-plus-diagonal alignment substantially recovered all three eligible degree-2 failures at 75% depth, while the MLP probe had no eligible failure there.

Operational metrics improved but did not reach the target-trained oracle. Across the six primary failures, mean TPR at the source 1% FPR threshold rose from 2.7% to 21.0%, while achieved FPR rose from 0.16% to 1.83%. Expected calibration error improved from 0.357 to 0.239, compared with 0.075 for the oracle.

These results identify a substantial, restricted coordinate component in this natural stage transition, but they do not prove an exact parameter-space symmetry: the map was learned from paired activations, residual performance and calibration gaps remain, and the positive result was localized to a late layer and one task. The next confirmatory step is to test independently initialized checkpoints and replicate the stage-transition result on a second task or model family.
