# August 27, 2026 | Progress Report and Way Forward

[Proposal](../docs/proposal.md) | [Frozen transfer](frozen_probe_transfer_baseline/report.md) | [Natural alignment](modern_activation_alignment_recovery/report.md) | [Residual symmetry](modern_residual_permutation_probe_transport/report.md) | [MLP symmetry](modern_mlp_neuron_permutation_probe_transport/report.md) | [Hugging Face artifacts](https://huggingface.co/buckets/baimamboukar/coordinate-concept-mismatch/tree/studies) | [Weights & Biases](https://wandb.ai/JinesisLab/coordinate-concept-mismatch)

## Progression of evidence

This project asks whether cross-model probe failure reflects incompatible internal coordinates or deeper representational differences.

The first causal control used exact residual permutations in Pythia. The transformations preserved model behavior, naïve transfer failed in every qualifying comparison, and analytic probe transport restored the original result. This established that coordinate mismatch is sufficient to cause probe failure, but only in a small-model setting.

The natural Pythia checkpoint comparison then tested whether restricted, label-free alignment could repair transfer between independently trained models. Permutation-diagonal alignment recovered a median 60.3% of the gap, while flexible linear maps recovered approximately 97% and a shuffled-pair control recovered none. Thus, natural mismatch was only partly consistent with the restricted symmetry.

The modern frozen-transfer baseline extended the question to Llama, Qwen, Mistral, Granite, and Nemotron. All 20 prespecified cross-family direction–seed comparisons failed, with a median AUROC gap of 0.411 and transfer TPR at 1% FPR of only 0–4.4%. By contrast, the Llama–Nemotron lineage control failed in 0/24 comparisons, demonstrating that transfer failure is not inevitable between distinct checkpoints.

Label-free alignment recovered a median 61.8% of the modern cross-family gap with a restricted permutation-diagonal map and 97.3% with flexible affine and quotient maps. The shuffled control passed 0/20 comparisons, while Granite $\rightarrow$ Qwen remained a clear directional failure. Flexible recovery establishes linear recoverability, not exact parameter symmetry.

Finally, we applied exact residual-coordinate permutations to Mistral-7B-v0.3, Llama-3.1-8B-Instruct, and Qwen3-8B. All nine full function gates passed with 100% next-token agreement. Naïve AUROC fell by 0.432–0.442 across models, producing coordinate-induced failure in all 36 probe comparisons. Analytic transport and label-free strict permutation estimation recovered all 36, and the planted mapping was identified exactly in all 12 model–seed combinations.

The MLP-neuron control then moved the intervention from the residual stream to Mistral's post-SwiGLU intermediate representation. Exact permutations preserved behavior on all 1,699 test prompts while naïve mean AUROC fell from 0.908 to 0.429. All 12 comparisons failed under naïve transfer; analytic transport and label-free permutation recovery restored all 12, with exact recovery of every planted map.

## Current conclusion and next step

Coordinate mismatch is sufficient to cause large probe-transfer failures and explains a substantial portion of natural cross-family gaps. It does not explain every restricted-alignment failure. The next controlled interventions should cover attention-component permutations, valid positive rescalings, and normalization-compatible transformations, then compare their recovery signatures with natural failures, especially Granite $\rightarrow$ Qwen. A second probe task and an unequal-width model pair should follow before making a broad concept-mismatch claim.
