# August 27, 2026 | Positive-Diagonal Scale Sweep

[Hugging Face artifacts](https://huggingface.co/buckets/baimamboukar/coordinate-concept-mismatch/tree/studies/positive-diagonal-scale-sweep/modern-mlp-positive-diagonal-scale-sweep/mistral-7b-v0.3) | [Weights & Biases](https://wandb.ai/JinesisLab/coordinate-concept-mismatch/runs/ulkk3auo)

## Summary

This experiment tested whether frozen-probe degradation increases with the magnitude of an exact, function-preserving positive-diagonal reparameterization. We held Mistral-7B-v0.3, its block-24 post-SwiGLU site, WildGuardMix splits, and the trained probes fixed. For transformation seeds 42 and 137, the same random directions were evaluated at mild $[1/2,2]$, moderate $[1/8,8]$, strong $[1/32,32]$, and extreme $[1/128,128]$ ranges. The analysis covered linear, degree-2 CP, and one-hidden-layer MLP probes across two data seeds.

All 9 smoke gates and all 9 full gates passed over 1,699 protected prompts. Next-token agreement was 100%; the maximum logit error was $3.15\times10^{-6}$. The intervention therefore changed the probed coordinates without materially changing model behavior.

## Results

| Scale range | Raw AUROC | AUROC gap | AUPRC | Balanced accuracy | TPR at 1% FPR | Failures |
|---|---:|---:|---:|---:|---:|---:|
| Reference | 0.908 | — | 0.898 | 0.827 | 0.327 | — |
| Mild | 0.906 | 0.003 | 0.895 | 0.829 | 0.328 | 0/12 |
| Moderate | 0.878 | 0.030 | 0.864 | 0.809 | 0.265 | 0/12 |
| Strong | 0.832 | 0.077 | 0.809 | 0.760 | 0.214 | 4/12 |
| Extreme | 0.799 | 0.109 | 0.773 | 0.727 | 0.183 | 4/12 |

The mean AUROC gap increased at every range, and all 12 paired trajectories had Spearman $\rho=1.0$. The prespecified dose-response criterion therefore passed. However, no range produced the required 10 of 12 coordinate failures: strong and extreme each produced only 4 of 12. All eight failures were degree-2 CP probes, covering every data/transformation-seed pair at those ranges; no linear or one-hidden-layer MLP probe crossed the rule. The preregistered robust-crossing outcome is consequently **not established**, even though the extreme mean gap exceeded 0.10.

Analytic transport and label-free diagonal estimation matched the reference scores in all 48 comparisons. Their thresholded recovery flag passed 47 of 48 because one mild transformation slightly improved raw AUROC, making a positive-gap recovery fraction undefined. Estimated scales matched every planted coordinate; maximum alignment relative RMSE was $4.81\times10^{-8}$.

## Interpretation

Probe fragility varies smoothly with the magnitude of this exact symmetry, but the effect is heterogeneous across probes. This strengthens the causal coordinate-mismatch result while showing that a mean effect cannot substitute for broadly replicated failure. The next experiment should test whether the dose response generalizes to a second task before extending the transformation family.
