# August 28, 2026 | OLMo 1 Cross-Task Alignment-Map Transport

[Plan](plan.md) | [SST-2→WildGuard artifacts](https://huggingface.co/buckets/baimamboukar/coordinate-concept-mismatch/tree/studies/olmo1-cross-task-map-transport/olmo1-map-transport-sst2-to-wildguard/sst2-fit-wildguard-eval/results) | [WildGuard→SST-2 artifacts](https://huggingface.co/buckets/baimamboukar/coordinate-concept-mismatch/tree/studies/olmo1-cross-task-map-transport/olmo1-map-transport-wildguard-to-sst2/wildguard-fit-sst2-eval/results) | [SST-2→WildGuard W&B](https://wandb.ai/JinesisLab/coordinate-concept-mismatch/runs/7bpk4d1v) | [WildGuard→SST-2 W&B](https://wandb.ai/JinesisLab/coordinate-concept-mismatch/runs/byzhkj0l)

## Summary

This experiment tested whether the dense Ai2↔AMD OLMo alignment learned on one unlabeled prompt distribution is a task-independent change of basis. Maps fitted on SST-2 training activations were applied to WildGuardMix probes and test activations without refitting; WildGuardMix→SST-2 transport was evaluated symmetrically. Models, tokenization, depths, probes, seeds, metrics, and same-task references were frozen.

Neither transport direction passed the preregistered affine-map criterion. SST-2→WildGuard recovered a median 28.7% of the AUROC gap, retained 30.1% of the same-task improvement, and was substantial in 1/4 comparisons. WildGuard→SST-2 recovered 37.4%, retained 37.9%, and was substantial in 2/4. Across both tests, median recovery was 36.4%, only 3/8 comparisons were substantial, and shuffled-pair maps were substantial in 0/8. This is sharply below the previous same-task affine recovery of 94.9% on WildGuardMix and 99.1% on SST-2.

The failure was direction- and task-dependent. Under SST-2→WildGuard transport, median recovery was 50.4% for Ai2→AMD but −10.6% for AMD→Ai2. Under WildGuard→SST-2 transport, the pattern reversed: 10.6% and 62.3%, respectively. A single bidirectional global coordinate map is therefore not supported.

Operational metrics showed partial but incomplete reuse. For WildGuardMix, median TPR at 1% FPR increased from 1.9% before alignment to 6.0% after cross-task affine alignment, versus 18.0% for the target-trained oracle; calibration error worsened from 0.269 to 0.392. For SST-2, TPR increased from 0.9% to 8.2%, versus 58.1% for the oracle, while calibration error remained 0.408 versus 0.033.

No secondary method established task-independent transport. Orthogonal recovery reached 52.2% in SST-2→WildGuard but passed only 2/4 comparisons; in the reverse test it recovered 33.4% and passed 0/4. Depth and probe-family effects remained heterogeneous.

The result narrows the earlier conclusion: dense paired alignment reveals task-local linear recoverability, but the fitted basis does not transport reliably across these two concepts. Next, a task-balanced pooled map should be fitted without labels and evaluated on a third held-out task. This will distinguish inadequate activation-support coverage from genuinely task-conditional model alignment.
