# August 27, 2026 | Progress Report and Way Forward

[Proposal](../docs/proposal.md) | [Frozen transfer](frozen_probe_transfer_baseline/report.md) | [Natural alignment](modern_activation_alignment_recovery/report.md) | [Residual symmetry](modern_residual_permutation_probe_transport/report.md) | [MLP permutation](modern_mlp_neuron_permutation_probe_transport/report.md) | [Attention symmetry](modern_attention_head_permutation_probe_transport/report.md) | [MLP rescaling](modern_mlp_positive_diagonal_probe_transport/report.md) | [Scale sweep](positive_diagonal_scale_sweep/report.md) | [Sentiment replication](sentiment_positive_diagonal_scale_sweep/report.md) | [Hugging Face artifacts](https://huggingface.co/buckets/baimamboukar/coordinate-concept-mismatch/tree/studies) | [Weights & Biases](https://wandb.ai/JinesisLab/coordinate-concept-mismatch)

## Progression of evidence

This project asks whether probe-transfer failure reflects incompatible coordinates or deeper representational differences.

Exact Pythia residual permutations preserved behavior, broke naive transfer, and were repaired by analytic transport, establishing that coordinate mismatch is sufficient to cause probe failure.

Natural Pythia checkpoints yielded 60.3% median recovery with permutation-diagonal alignment, about 97% with flexible maps, and none with shuffled pairs; restricted symmetry explained only part of the mismatch.

Across Llama, Qwen, Mistral, Granite, and Nemotron, all 20 cross-family comparisons failed: median AUROC gap was 0.411 and 1%-FPR TPR was 0-4.4%. The Llama-Nemotron lineage control failed in 0/24 comparisons.

Label-free alignment recovered a median 61.8% of the modern cross-family gap with permutation-diagonal maps and 97.3% with flexible affine and quotient maps. The shuffled control passed 0/20 comparisons, while Granite to Qwen remained a directional failure. Flexible recovery establishes linear recoverability, not exact parameter symmetry.

Exact modern residual permutations across Mistral, Llama, and Qwen preserved behavior and induced failure in all 36 comparisons; analytic and strict label-free transport recovered all 36. The component-local Mistral MLP control induced a mean AUROC drop from 0.908 to 0.429, with exact recovery in all 12 comparisons.

The GQA attention-head control extended this result to the pre-output-projection representation. All full gates passed with 100% next-token agreement. Mean AUROC fell from 0.889 to 0.660 under naive transfer; all 12 comparisons failed, while analytic and label-free transport recovered all 12 and exactly recovered every planted map.

The positive-diagonal MLP control then held the model, site, data, and probes fixed while changing the symmetry type. Mean AUROC fell only from 0.908 to 0.878; none of 12 comparisons met the prespecified 0.10 failure threshold, although analytic and label-free transport recovered all 12. This contrast shows that exact symmetries are not uniformly disruptive: transformation geometry and magnitude affect probe fragility.

The WildGuardMix scale sweep resolved the magnitude question: mean gaps rose monotonically from 0.003 to 0.109, with failures confined to degree-2 CP probes at strong and extreme scaling. Exact and estimated transport matched all 48 reference scores.

The targeted SST-2 replication retained the model, site, ranges, and probes while changing the task. Reference AUROC was 0.973-0.979. Mean gaps again rose monotonically, from 0.001 to 0.106. CP probes failed in 3/4 strong and 4/4 extreme comparisons, versus 0/8 linear/MLP failures at each range; the preregistered family-sensitivity rule passed. Transport matched reference scores in all 48 comparisons. This establishes out-of-task replication, not cross-model generality.

## Current conclusion and next step

Coordinate mismatch is sufficient to cause large failures, but its impact depends on geometry, magnitude, and probe architecture. Next, test a scale-normalized degree-2 probe to separate nonlinear expressivity from numerical conditioning.
