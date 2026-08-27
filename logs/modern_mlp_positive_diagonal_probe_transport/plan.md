# August 27, 2026 | MLP Positive-Diagonal Probe Transport

[Hugging Face artifacts](https://huggingface.co/buckets/baimamboukar/coordinate-concept-mismatch/tree/studies/modern-mlp-positive-diagonal-probe-transport/modern-mlp-positive-diagonal-symmetry) | [Weights & Biases](https://wandb.ai/JinesisLab/coordinate-concept-mismatch)

## Objective

The permutation controls established that reordering residual, MLP, and attention coordinates can break probe transfer without changing model behavior. This experiment holds the model, representation, data, and probes fixed while testing a different symmetry group: independent positive rescaling of Mistral's post-SwiGLU MLP coordinates. It asks whether probe failure depends on the type and magnitude of a coordinate change rather than permutation alone.

## Formal setup

At transformer block $\ell$, Mistral's MLP is

$$
h_\ell(x)=\operatorname{SiLU}(W_{\mathrm{gate},\ell}x)\odot W_{\mathrm{up},\ell}x,
\qquad
m_\ell(x)=W_{\mathrm{down},\ell}h_\ell(x).
$$

For a positive diagonal matrix $D$, define

$$
W_{\mathrm{up},\ell}^{D}=DW_{\mathrm{up},\ell},
\qquad
W_{\mathrm{down},\ell}^{D}=W_{\mathrm{down},\ell}D^{-1}.
$$

Then $h_\ell^{D}(x)=Dh_\ell(x)$ while $m_\ell^{D}(x)=m_\ell(x)$ exactly. The gate projection is unchanged; scaling it would not generally be a symmetry because SiLU is not positively homogeneous.

## Protocol

- **Model and site:** pinned Mistral-7B-v0.3, block 24 of 32, captured after SwiGLU and before `down_proj`.
- **Materials:** reuse the verified WildGuardMix activations and linear, degree-2 CP, and one-hidden-layer MLP probes from the MLP-permutation baseline; data seeds 42 and 137.
- **Interventions:** two seeded diagonal maps with 14,336 independently sampled log-uniform scales in $[1/8,8]$.
- **Comparisons:** reference, identity, naive transfer, analytic probe transport, label-free positive-diagonal estimation from paired training activations, and inverse transport.
- **Function gate:** eight-prompt fail-fast followed by all 1,699 protected test prompts in FP64. Every map must preserve logits within tolerance, retain 100% next-token agreement, and satisfy $h_\ell^{D}=Dh_\ell$.

## Decision rule

Primary outcomes are the raw AUROC gap and analytic and estimated recovery fractions. Coordinate-induced failure requires reference AUROC at least 0.75, a gap of at least 0.10, and a paired 95% bootstrap interval excluding zero; recovery succeeds at 95%. Secondary outputs retain AUPRC, accuracy, balanced accuracy, precision, recall, F1, calibration, confusion counts, thresholds, TPR at 1% and 5% FPR, scale-estimation diagnostics, and row-level predictions.

A passed result establishes sufficiency for an exact positive-diagonal mismatch at one MLP site. It does not show that natural cross-model differences are diagonal or that the effect generalizes to other tasks or layers.
