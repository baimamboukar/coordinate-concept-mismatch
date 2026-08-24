# August 24th, 2026 | Pythia Residual-Permutation Probe Transport

[Hugging Face artifacts](https://huggingface.co/buckets/baimamboukar/coordinate-concept-mismatch/tree/experiments/pythia_residual_permutation_probe_transport) | [Weights & Biases](https://wandb.ai/JinesisLab/coordinate-concept-mismatch/runs/nlvrx25u) | [Experiment plan](plan.md)

## Summary

This experiment tested whether a known residual-coordinate permutation can cause probe-transfer failure while preserving the language model's function. We evaluated `EleutherAI/pythia-410m` and `pythia-410m-seed1`, data seeds 42 and 137, permutation seeds 42 and 137, and linear, degree-2, and MLP probes at block 18.

All six full-test-set function gates passed in FP64 across 1,699 prompts. Maximum logit error was $1.14\times10^{-11}$, maximum activation-equivariance error was $1.27\times10^{-13}$, and next-token agreement was 100%.

Across all 24 primary comparisons, reference AUROC was 0.834–0.877 and raw-permutation AUROC was 0.440–0.632. The AUROC gap was 0.239–0.425, with median 0.321; every paired 95% bootstrap interval excluded zero. Exact probe transport restored every comparison to its reference AUROC with recovery fraction 1.00 and maximum score error $2.10\times10^{-5}$. Inverse-direction transport did not recover performance.

Two preliminary FP32 gates stopped before probe evaluation because accumulation-order differences exceeded the initial logit tolerances despite 100% next-token agreement. The final FP64 certification isolates the exact symmetry from this numerical effect.

## Interpretation

Residual-coordinate mismatch alone is sufficient to produce severe frozen-probe failure in a functionally equivalent model, and analytically correct probe transport removes that failure. This does not establish how much of the natural gap between independently trained checkpoints is coordinate-based; the next experiment must estimate that component through checkpoint alignment.
