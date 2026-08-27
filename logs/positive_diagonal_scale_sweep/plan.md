# August 27, 2026 | Positive-Diagonal Scale Sweep

[Hugging Face artifacts](https://huggingface.co/buckets/baimamboukar/coordinate-concept-mismatch/tree/studies/positive-diagonal-scale-sweep/modern-mlp-positive-diagonal-scale-sweep) | [Weights & Biases](https://wandb.ai/JinesisLab/coordinate-concept-mismatch)

## Objective

The preceding positive-diagonal control found a small but systematic AUROC loss under scales in $[1/8,8]$, below the prespecified 0.10 failure threshold. This follow-up estimates whether probe degradation increases with transformation magnitude and identifies the first range producing robust coordinate-induced failure. The $[1/8,8]$ condition is an internal replication of an observed result; all other ranges and the dose-response rules below were fixed before evaluating them.

## Formal setup

For Mistral block $\ell$, let $h_\ell(x)$ denote the post-SwiGLU activation and $W_{\mathrm{down},\ell}$ its output projection. For positive diagonal $D_r$ with log-radius $r$,

$$
h_\ell^{D_r}(x)=D_rh_\ell(x),
\qquad
W_{\mathrm{down},\ell}^{D_r}=W_{\mathrm{down},\ell}D_r^{-1}.
$$

Thus the probed coordinates change while the MLP output and model function remain invariant. For each transformation seed, the same underlying uniform draws define four paired ranges: mild $[1/2,2]$, moderate $[1/8,8]$, strong $[1/32,32]$, and extreme $[1/128,128]$.

## Protocol

- **Materials:** pinned Mistral-7B-v0.3; block 24 post-SwiGLU activations; WildGuardMix data seeds 42 and 137; existing linear, degree-2 CP, and one-hidden-layer MLP probes.
- **Interventions:** four ordered scale ranges and transformation seeds 42 and 137, giving 12 paired probe trajectories and 48 recovery comparisons.
- **Controls:** reference, identity, naive transfer, analytic transport, label-free diagonal estimation, and inverse transport.
- **Function gate:** eight-prompt fail-fast followed by all 1,699 protected test prompts in FP64. Every condition must preserve logits, next-token choices, and the planted activation transformation.

## Decision rules

The primary outcomes are mean raw AUROC gap by range, dose-response support, first robust threshold crossing, and analytic and estimated recovery. Dose response requires nondecreasing range-level mean gaps and Spearman $\rho\geq0.8$ for at least 10 of 12 paired trajectories. A robust crossing is the earliest range where at least 10 of 12 comparisons satisfy the existing rule: reference AUROC at least 0.75, gap at least 0.10, and paired 95% bootstrap lower bound above zero. Recovery succeeds at 95%. Secondary reporting retains AUPRC, accuracy, balanced accuracy, precision, recall, F1, calibration, confusion counts, thresholds, TPR at 1% and 5% FPR, scale diagnostics, and row-level predictions.
