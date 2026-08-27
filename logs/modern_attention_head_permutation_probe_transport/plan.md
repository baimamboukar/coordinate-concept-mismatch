# August 27, 2026 | Attention-Head Permutation Probe Transport

[Hugging Face artifacts](https://huggingface.co/buckets/baimamboukar/coordinate-concept-mismatch/tree/studies/modern-attention-head-permutation-probe-transport/modern-attention-head-symmetry) | [Weights & Biases](https://wandb.ai/JinesisLab/coordinate-concept-mismatch)

## Objective

The residual and MLP controls showed that exact coordinate changes can break probes at global and component-local representations. This experiment tests a structurally constrained symmetry of grouped-query attention (GQA): head permutations that preserve each query head's key/value association. The aim is to determine whether naïve probe transfer fails at the concatenated attention-head output and whether analytic or activation-estimated transport restores it.

## Formal setup

At transformer block $\ell$, let $H_Q$ query heads share $H_{KV}$ key/value heads, with group size $G=H_Q/H_{KV}$. Let $P_Q$ permute query heads by jointly permuting GQA groups and optionally reordering the $G$ query heads within each group. Let $P_{KV}$ be the induced key/value-group permutation. We transform

$$
W_{Q,\ell}^{P}=P_QW_{Q,\ell},\qquad
W_{K,\ell}^{P}=P_{KV}W_{K,\ell},\qquad
W_{V,\ell}^{P}=P_{KV}W_{V,\ell},\qquad
W_{O,\ell}^{P}=W_{O,\ell}P_Q^{\top}.
$$

The concatenated pre-output-projection representation becomes $a_\ell^{P}(x)=P_Qa_\ell(x)$, while the attention block output, residual stream, logits, and predictions remain unchanged.

## Protocol

- **Model and site:** pinned Mistral-7B-v0.3, block 24 of 32, captured immediately before `self_attn.o_proj`; 32 query heads, eight key/value heads, head dimension 128.
- **Data:** WildGuardMix with 12,000 training, 2,000 validation, and 1,699 protected test examples; data seeds 42 and 137.
- **Probes:** linear, degree-2 CP, and one-hidden-layer MLP probes.
- **Interventions:** structured head permutations seeded by 42 and 137.
- **Comparisons:** reference, identity, naïve transfer, analytic transport, label-free exact feature matching, and inverse-transport control.
- **Function gate:** eight-prompt fail-fast followed by all 1,699 test prompts in FP64. Every transformation must preserve logits within tolerance, retain 100% next-token agreement, and satisfy attention-output equivariance.

## Decision rule

Primary outcomes are the raw AUROC gap, analytic recovery fraction, and activation-estimated recovery fraction. Coordinate-induced failure requires reference AUROC at least 0.75, an AUROC gap of at least 0.10, and a paired 95% bootstrap interval excluding zero. Recovery succeeds at 95%. Secondary outputs retain AUPRC, accuracy, balanced accuracy, precision, recall, F1, calibration, confusion counts, thresholds, TPR at 1% and 5% FPR, alignment diagnostics, and row-level predictions.

A passed result establishes sufficiency for an exact attention-local coordinate mismatch; it does not imply that natural cross-model attention representations differ only by head permutation.
