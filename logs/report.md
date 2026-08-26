# August 25th, 2026 | Progress Report and Way Forward

[Proposal](../docs/proposal.md) | [Frozen-transfer report](frozen_probe_transfer_baseline/report.md) | [Alignment report](modern_activation_alignment_recovery/report.md) | [Hugging Face artifacts](https://huggingface.co/buckets/baimamboukar/coordinate-concept-mismatch/tree/studies) | [Weights & Biases](https://wandb.ai/JinesisLab/coordinate-concept-mismatch)

## What we established

This project asks whether cross-model probe failure reflects a change of internal coordinates or a deeper difference in learned representations. We have built a gradual experimental chain that moves from a causal symmetry control to independently trained modern model families.

First, the controlled Pythia residual-permutation experiment showed that coordinate mismatch alone can cause severe probe failure. The transformed models preserved their function across all 1,699 test prompts, while naive probe transfer lost 0.239–0.425 AUROC in all 24 comparisons. Analytic probe transport restored every result exactly. This is a causal pipeline control, not yet modern-model paper evidence.

Second, alignment between independently trained Pythia checkpoints recovered a median 60.3% of the natural transfer gap with the restricted permutation-diagonal map; 11/12 comparisons passed. Flexible linear maps recovered approximately 97%, whereas the shuffled-pair control passed 0/12.

Third, the modern frozen-transfer baseline established broad failure across Llama, Qwen, Mistral, and Granite. All 20 prespecified primary direction-seed comparisons failed, with a median AUROC gap of 0.411. Transfer TPR at 1% FPR was only 0–4.4%, compared with 31.2–48.1% for target-trained probes. In contrast, the Llama–Nemotron lineage control failed in 0/24 comparisons, showing that transfer failure is not inevitable between distinct checkpoints.

Finally, label-free permutation-diagonal alignment recovered a median 61.8% of the modern cross-family gap and passed 16/20 comparisons. Flexible affine and quotient maps recovered 97.3%, while the shuffled control passed 0/20. Granite $\rightarrow$ Qwen remained the clearest directional failure.

## Current conclusion

The evidence supports a mixed account. Coordinate mismatch is sufficient to cause failure and explains a substantial portion of natural transfer gaps. However, restricted alignment does not repair every pair. Near-complete flexible recovery establishes linear recoverability, not an exact parameter symmetry, and the remaining gap is unexplained rather than proof of concept mismatch.

## Way forward

The immediate priority is to reproduce the exact function-preserving symmetry study on a modern transformer, beginning with Mistral-7B-v0.3. We should test valid MLP-neuron permutations, attention-component permutations, positive rescalings, and normalization-compatible transformations. Each intervention must pass strict logit, next-token, and activation-equivariance gates before comparing naive transfer, analytic probe transport, and activation-estimated alignment.

Next, we should diagnose Granite $\rightarrow$ Qwen across layers, alignment-set sizes, and restricted map classes while preserving label-free fitting and the protected test set. Only after adding a second probe task or dataset and an unequal-width family should we make broader claims about concept mismatch or architectural generality.
