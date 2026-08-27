# August 27, 2026 | MLP Positive-Diagonal Probe Transport

[Hugging Face artifacts](https://huggingface.co/buckets/baimamboukar/coordinate-concept-mismatch/tree/studies/modern-mlp-positive-diagonal-probe-transport/modern-mlp-positive-diagonal-symmetry) | [Weights & Biases](https://wandb.ai/JinesisLab/coordinate-concept-mismatch/runs/9j2555iv)

## Objective and method

This experiment tested whether an exact positive rescaling of Mistral's post-SwiGLU coordinates is sufficient to cause probe-transfer failure. It reused the pinned Mistral-7B-v0.3 checkpoint, block-24 activations, WildGuardMix splits, and linear, degree-2 CP, and one-hidden-layer MLP probes from the MLP-permutation control. Data seeds were 42 and 137, with 12,000 training, 2,000 validation, and 1,699 protected test examples.

For two transformation seeds, 14,336 independent scales were sampled log-uniformly from $[1/8,8]$. The up-projection rows were multiplied by a positive diagonal matrix $D$, and the corresponding down-projection columns by $D^{-1}$. This rescales the probed representation while preserving the MLP output and model function. We compared naive transfer, analytic probe transport, label-free diagonal estimation from paired activations, and an inverse-transport control.

## Results

All three full function gates passed. Next-token agreement was 100%, maximum logit error was $2.51\times10^{-6}$, and maximum activation-equivariance error was $1.87\times10^{-15}$.

| Condition | Mean AUROC | Mean AUPRC | Mean balanced accuracy | Mean TPR at 1% FPR |
| --- | ---: | ---: | ---: | ---: |
| Reference | 0.908 | 0.898 | 0.827 | 0.327 |
| Naive transfer | 0.878 | 0.864 | 0.809 | 0.265 |
| Analytic transport | 0.908 | 0.898 | 0.827 | 0.327 |
| Activation-estimated alignment | 0.908 | 0.898 | 0.827 | 0.327 |
| Inverse control | 0.813 | 0.788 | 0.740 | 0.197 |

The mean raw AUROC gap was 0.030, ranging from 0.006 to 0.075. Every paired 95% bootstrap interval excluded zero, but none of the 12 comparisons reached the prespecified 0.10 failure threshold. Analytic transport and activation-estimated alignment recovered all 12 comparisons. All four estimated maps matched every planted scale coordinate; maximum relative scale error was $1.23\times10^{-7}$. Public artifacts retain all secondary metrics, thresholds, confusion counts, calibration values, diagnostics, and 101,940 row-level predictions.

## Interpretation and next step

This is a controlled negative result: moderate positive-diagonal mismatch causes a small, systematic degradation, but not the substantial transfer failure produced by permutations at the same MLP site. Exact symmetry alone therefore does not determine probe fragility; transformation geometry and magnitude matter. The next experiment should preregister a scale-range sweep to estimate this sensitivity curve before testing normalization-compatible transformations or a second probe task.
