# August 27, 2026 | Progress Report and Way Forward

[Proposal](../docs/proposal.md) | [Frozen transfer](frozen_probe_transfer_baseline/report.md) | [Natural alignment](modern_activation_alignment_recovery/report.md) | [Residual symmetry](modern_residual_permutation_probe_transport/report.md) | [MLP permutation](modern_mlp_neuron_permutation_probe_transport/report.md) | [Attention symmetry](modern_attention_head_permutation_probe_transport/report.md) | [MLP rescaling](modern_mlp_positive_diagonal_probe_transport/report.md) | [Hugging Face artifacts](https://huggingface.co/buckets/baimamboukar/coordinate-concept-mismatch/tree/studies) | [Weights & Biases](https://wandb.ai/JinesisLab/coordinate-concept-mismatch)

## Progression of evidence

This project asks whether cross-model probe failure reflects incompatible internal coordinates or deeper representational differences.

The first causal control used exact residual permutations in Pythia. Model behavior was preserved, naive transfer failed, and analytic transport restored the original result, establishing that coordinate mismatch is sufficient to cause probe failure in a small-model setting.

Natural Pythia checkpoint comparisons then tested restricted, label-free alignment between independently trained models. Permutation-diagonal alignment recovered a median 60.3% of the gap, flexible linear maps recovered approximately 97%, and a shuffled-pair control recovered none. Natural mismatch was therefore only partly consistent with the restricted symmetry.

The modern frozen-transfer baseline covered Llama, Qwen, Mistral, Granite, and Nemotron. All 20 prespecified cross-family direction-seed comparisons failed, with a median AUROC gap of 0.411 and transfer TPR at 1% FPR of 0-4.4%. The Llama-Nemotron lineage control failed in 0/24 comparisons, showing that failure is not inevitable between distinct checkpoints.

Label-free alignment recovered a median 61.8% of the modern cross-family gap with permutation-diagonal maps and 97.3% with flexible affine and quotient maps. The shuffled control passed 0/20 comparisons, while Granite to Qwen remained a directional failure. Flexible recovery establishes linear recoverability, not exact parameter symmetry.

Exact modern residual permutations across Mistral, Llama, and Qwen preserved behavior and induced failure in all 36 comparisons; analytic and strict label-free transport recovered all 36. The component-local Mistral MLP control induced a mean AUROC drop from 0.908 to 0.429, with exact recovery in all 12 comparisons.

The GQA attention-head control extended this result to the pre-output-projection representation. All full gates passed with 100% next-token agreement. Mean AUROC fell from 0.889 to 0.660 under naive transfer; all 12 comparisons failed, while analytic and label-free transport recovered all 12 and exactly recovered every planted map.

The positive-diagonal MLP control then held the model, site, data, and probes fixed while changing the symmetry type. Mean AUROC fell only from 0.908 to 0.878; none of 12 comparisons met the prespecified 0.10 failure threshold, although analytic and label-free transport recovered all 12. This contrast shows that exact symmetries are not uniformly disruptive: transformation geometry and magnitude affect probe fragility.

## Current conclusion and next step

Coordinate mismatch is sufficient to cause large failures, but an exact coordinate change need not do so. Next, preregister a positive-scale magnitude sweep, then test normalization-compatible transformations and a second probe task before making a broad concept-mismatch claim.
