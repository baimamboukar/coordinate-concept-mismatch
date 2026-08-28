# August 28, 2026 | Held-Out BoolQ Qualification

[Plan](plan.md) | [Hugging Face activations](https://huggingface.co/buckets/baimamboukar/coordinate-concept-mismatch/tree/activations/boolq-qa-v1) | [Hugging Face baseline](https://huggingface.co/buckets/baimamboukar/coordinate-concept-mismatch/tree/studies/olmo1-independent-probe-transfer/olmo1-independent-training-boolq/boolq-baseline) | [Weights & Biases](https://wandb.ai/JinesisLab/coordinate-concept-mismatch/runs/q03g7gtt)

## Summary

The preceding SST-2↔WildGuardMix experiment found that dense alignment maps recover same-task probe transfer but transport poorly across tasks. This experiment was designed to test whether broader activation coverage solves that problem: fit equal-budget maps on SST-2, WildGuardMix, or their balanced pool, then evaluate them without refitting on held-out BoolQ.

We first applied the preregistered qualification gate. Frozen probes were trained and evaluated between the pinned Ai2 and AMD OLMo 1B checkpoints at 75% residual depth, using seeds 42 and 137. The protected BoolQ validation split was used only for testing. Linear probes were primary; degree-2 and MLP probes were retained as secondary families, together with the full metric and row-level prediction contract.

All four primary linear comparisons exhibited large cross-model AUROC gaps, ranging from 0.153 to 0.166. Their paired 95% confidence intervals excluded zero. However, target-trained linear AUROC ranged only from 0.639 to 0.646, below the prespecified 0.75 oracle threshold. Consequently, none of the four comparisons satisfied the complete probe-transfer-failure rule.

BoolQ therefore failed the qualification gate. In accordance with the plan, we did not fit the same-task, single-task, or pooled alignment maps, and we make no claim about their held-out transport performance. This is not evidence that pooled alignment fails: the experiment was stopped because the selected representation was insufficiently linearly predictive of BoolQ truth to support the intended recovery analysis.

The next experiment should be separately preregistered around a third task that first satisfies a task-independent decodability gate. BoolQ remains a documented non-qualifying task and will not be replaced post hoc within this experiment.
