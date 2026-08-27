# August 27, 2026 | MLP-Neuron Permutation Probe Transport

[Hugging Face artifacts](https://huggingface.co/buckets/baimamboukar/coordinate-concept-mismatch/tree/studies/modern-mlp-neuron-permutation-probe-transport/modern-mlp-neuron-symmetry) | [W&B baseline](https://wandb.ai/JinesisLab/coordinate-concept-mismatch/runs/d0if1kcx) | [W&B symmetry](https://wandb.ai/JinesisLab/coordinate-concept-mismatch/runs/uhjukf5m)

## Objective and method

This experiment tested whether an exact, component-local coordinate change is sufficient to break probe transfer. We extracted the post-SwiGLU MLP representation at block 24 of the pinned Mistral-7B-v0.3 checkpoint. Linear, degree-2 CP, and one-hidden-layer MLP probes were trained on WildGuardMix using data seeds 42 and 137, with 12,000 training, 2,000 validation, and 1,699 protected test examples.

For permutation seeds 42 and 137, we jointly permuted the output neurons of the gate and up projections and the corresponding input columns of the down projection. This preserves the MLP output and model function while permuting the probed representation. We compared naïve transfer, analytic probe transport, label-free permutation recovery from paired activations, and an inverse-transport control.

## Results

All three full function gates passed on the complete test set. Next-token agreement was 100%, maximum logit error was $2.02\times10^{-6}$, and measured activation-equivariance error was zero.

| Condition | Mean AUROC | Mean balanced accuracy | Mean TPR at 1% FPR |
| --- | ---: | ---: | ---: |
| Reference | 0.908 | 0.827 | 0.327 |
| Naïve transfer | 0.429 | 0.474 | 0.008 |
| Analytic transport | 0.908 | 0.827 | 0.327 |
| Activation-estimated alignment | 0.908 | 0.827 | 0.327 |
| Inverse control | 0.479 | 0.489 | 0.013 |

The mean raw AUROC gap was 0.480, ranging from 0.433 to 0.524. All 12 probe-family, data-seed, and permutation-seed comparisons met the prespecified coordinate-failure rule, with every paired 95% bootstrap interval excluding zero. Analytic transport and activation-estimated alignment recovered 12/12 comparisons. All four label-free alignment maps identified the planted permutation exactly. The public artifacts retain all prespecified secondary metrics, thresholds, confusion counts, calibration values, and row-level predictions.

## Interpretation and next step

Coordinate mismatch is therefore sufficient to cause severe transfer failure at a local MLP representation, not only in the residual stream. This controlled result does not imply that independently trained models differ only by neuron permutation. The next experiment should test attention-component permutations and valid positive rescalings, then compare their recovery signatures with the natural cross-model failures already observed.
