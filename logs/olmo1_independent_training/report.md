# August 28, 2026 | OLMo 1 Independent-Training Transfer

[Plan](plan.md) | [Baseline artifacts](https://huggingface.co/buckets/baimamboukar/coordinate-concept-mismatch/tree/studies/olmo1-independent-probe-transfer/olmo1-independent-training-sst2/sst2-baseline) | [Alignment artifacts](https://huggingface.co/buckets/baimamboukar/coordinate-concept-mismatch/tree/studies/olmo1-independent-alignment/olmo1-independent-training-sst2/sst2-alignment) | [Transfer W&B](https://wandb.ai/JinesisLab/coordinate-concept-mismatch/runs/emqp30a1) | [Alignment W&B](https://wandb.ai/JinesisLab/coordinate-concept-mismatch/runs/h6utqis7)

## Summary

This experiment tested bidirectional SST-2 probe transfer between architecture-identical OLMo 1B checkpoints trained independently by Ai2 and AMD. A shared pinned tokenizer and identical tokenization settings removed input-token differences. Because provider, corpus version, hardware, and token budget also differ, this is an independent-training comparison rather than a pure initialization intervention.

The preregistered transfer-failure criterion passed decisively. All four 75%-depth linear-probe comparisons failed: target-trained AUROC ranged from 0.946 to 0.968, transferred AUROC from 0.424 to 0.618, and the median transfer gap was 0.406. Thus, identical architecture and strong within-model probes were insufficient for direct cross-model reuse.

The primary restricted-recovery criterion did not pass. Permutation-plus-positive-diagonal alignment produced a median AUROC improvement of 0.135 and recovered 31.9% of the gap, with substantial recovery in only 1/4 comparisons. Median residual gap remained 0.265. Recovery was asymmetric: approximately 10% for Ai2→AMD and 59% for AMD→Ai2. Exact permutation was also inconsistent, recovering a median 35.7% with substantial recovery in 2/4 comparisons.

Dense alignments produced a different result. Orthogonal Procrustes recovered a median 92.5% of the gap, while affine and quotient maps recovered approximately 99.1%; all passed in 4/4 comparisons. Shuffled-pair affine alignment recovered none and reduced performance, showing that recovery depended on correct cross-model example correspondence rather than map capacity alone. At the operational 1% FPR point, median TPR was 0.9% before alignment, 8.2% after restricted alignment, 54.2% after affine alignment, and 58.1% for the target-trained oracle. Affine calibration error (0.030) also approached the oracle (0.033), whereas restricted alignment remained worse (0.120).

Secondary analyses showed that restricted recovery increased at the final depth (median 52.5%; 3/4 substantial) but remained below threshold at earlier depths. Degree-2 and MLP probes showed median restricted recovery of 47.1% and 48.7%, respectively.

The result extends natural probe-transfer failure to independently trained, architecture-identical models, but rejects the stronger claim that neuron permutation and rescaling generally explain it. The near-complete dense-map recovery instead shows that the task signal remains linearly recoverable under a broader basis change; this is evidence of coordinate mismatch, not proof of an exact parameter-space symmetry. The next test should repeat this contrast on a safety-relevant task using the same model pair.
