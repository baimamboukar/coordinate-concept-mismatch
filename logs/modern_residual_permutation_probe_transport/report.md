# August 26–27, 2026 | Modern Residual-Permutation Probe Transport

[Hugging Face artifacts](https://huggingface.co/buckets/baimamboukar/coordinate-concept-mismatch/tree/studies/modern-residual-permutation-probe-transport/modern-models) | W&B: [Mistral](https://wandb.ai/JinesisLab/coordinate-concept-mismatch/runs/mh826137) · [Llama](https://wandb.ai/JinesisLab/coordinate-concept-mismatch/runs/ml642818) · [Qwen](https://wandb.ai/JinesisLab/coordinate-concept-mismatch/runs/mq905369)

## Objective and method

We tested whether a known change of residual coordinates can independently cause frozen-probe transfer failure in three modern transformer families. For Mistral-7B-v0.3, Llama-3.1-8B-Instruct, and Qwen3-8B, we applied two seeded global residual-coordinate permutations while consistently transforming every affected parameter. We evaluated linear, degree-2 CP, and one-hidden-layer MLP probes at 75% depth using data seeds 42 and 137.

Each intervention first had to preserve the model's function and satisfy activation equivariance on all 1,699 protected test prompts. We then compared naïve transfer with analytic probe transport and a strict feature permutation estimated only from paired, unlabeled activations.

## Results

| Model | Function gates | Reference AUROC | Naïve AUROC | Mean gap | Analytic recovery | Estimated recovery |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Mistral-7B-v0.3 | 3/3 | 0.912 | 0.469 | 0.442 | 12/12 | 12/12 |
| Llama-3.1-8B-Instruct | 3/3 | 0.930 | 0.498 | 0.432 | 12/12 | 12/12 |
| Qwen3-8B | 3/3 | 0.928 | 0.489 | 0.440 | 12/12 | 12/12 |

All transformed models retained 100% next-token agreement. Maximum activation-equivariance error was below $1.74\times10^{-7}$ across models. All 36 probe comparisons met the prespecified coordinate-failure rule, and both analytic transport and estimated alignment recovered the reference performance in all 36. The estimated alignment also identified the exact planted mapping in all 12 model–seed combinations.

The public artifacts retain the complete secondary-metric contract—AUPRC, accuracy, balanced accuracy, precision, recall, F1, calibration, confusion counts, thresholds, and low-FPR TPR—together with 305,820 row-level predictions.

## Conclusion

Across Mistral, Llama, and Qwen, residual-coordinate mismatch alone is sufficient to produce severe linear and nonlinear probe-transfer failure without changing model behavior. A label-free restricted alignment can recover this failure when the true mismatch belongs to its assumed symmetry class. This causal result does not imply that natural cross-model gaps are exact permutations; it provides the positive control required to test that hypothesis.
