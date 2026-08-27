# August 27, 2026 | Sentiment Positive-Diagonal Scale Sweep

[Plan](plan.md) | [Hugging Face artifacts](https://huggingface.co/buckets/baimamboukar/coordinate-concept-mismatch/tree/studies/positive-diagonal-task-generalization/sentiment-positive-diagonal-scale-sweep) | [Baseline W&B run](https://wandb.ai/JinesisLab/coordinate-concept-mismatch/runs/7pledwon) | [Symmetry W&B run](https://wandb.ai/JinesisLab/coordinate-concept-mismatch/runs/qeea9ll0)

## Summary

This targeted replication tested whether the degree-2 probe sensitivity observed on WildGuardMix generalizes to sentence sentiment. Linear, degree-2 CP, and one-hidden-layer MLP probes were trained on Mistral-7B-v0.3 block-24 post-SwiGLU activations from balanced SST-2 splits for seeds 42 and 137, then evaluated on the protected 872-example test set under four function-preserving positive-diagonal scale ranges.

Task viability passed: all six reference AUROCs were between 0.9731 and 0.9793. All nine full function gates passed; maximum logit error was $1.22\times10^{-6}$ and maximum activation-relative error was $1.73\times10^{-15}$.

Mean raw AUROC gaps increased monotonically from 0.0015 (mild) to 0.0150 (moderate), 0.0579 (strong), and 0.1062 (extreme). All 12 probe-seed trajectories satisfied the preregistered Spearman criterion, so the dose-response hypothesis passed. The inherited overall failure rule did not pass: failures were 0/12, 0/12, 3/12, and 4/12 across the four ranges, never reaching 10/12.

The primary family-specific replication passed at both required ranges. At strong scaling, degree-2 CP probes failed in 3/4 comparisons with mean gap 0.1336, versus 0/8 failures for linear and MLP probes and a largest comparison-family mean gap of 0.0287. At extreme scaling, CP failed in 4/4 comparisons with mean gap 0.2549, versus 0/8 and 0.0453. The corresponding CP gap advantages were 0.1049 and 0.2096, both above the preregistered 0.05 margin.

Analytic and label-free estimated transport matched reference scores in all 48 comparisons, with maximum score error $4.77\times10^{-6}$. One mild MLP case had a slightly negative raw gap, so its recovery fraction was undefined rather than failed. The published evidence includes 204 symmetry metric rows, 177,888 row-level predictions, 48 recovery rows, and 16 alignment diagnostics.

## Conclusion

Degree-2 scale sensitivity generalizes from safety classification to sentiment on the same model. The effect is therefore not specific to WildGuardMix, but it remains probe-family selective rather than a universal consequence of diagonal symmetry. The next experiment should test a scale-normalized degree-2 probe to distinguish representational sensitivity from numerical conditioning.
