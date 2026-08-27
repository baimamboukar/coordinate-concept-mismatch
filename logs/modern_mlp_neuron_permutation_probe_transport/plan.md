# August 27, 2026 | MLP-Neuron Permutation Probe Transport

[Hugging Face artifacts](https://huggingface.co/buckets/baimamboukar/coordinate-concept-mismatch/tree/studies/modern-mlp-neuron-permutation-probe-transport/modern-mlp-neuron-symmetry) | [Weights & Biases](https://wandb.ai/JinesisLab/coordinate-concept-mismatch)

## Objective

The residual-coordinate experiment established that a global change of residual basis can destroy probe transfer while preserving model behavior. This experiment asks whether the same causal mechanism appears at a component-local representation: the intermediate neurons of a SwiGLU MLP. It also checks whether linear and nonlinear probes respond differently to the same exact coordinate change.

## Formal setup

At transformer block $\ell$, define the MLP intermediate representation

$$
z_\ell(x)=\operatorname{SiLU}(W_{g,\ell}x)\odot W_{u,\ell}x,
$$

with block output $W_{d,\ell}z_\ell(x)$. For a permutation matrix $P$, construct

$$
W_{g,\ell}^{P}=PW_{g,\ell},\qquad
W_{u,\ell}^{P}=PW_{u,\ell},\qquad
W_{d,\ell}^{P}=W_{d,\ell}P^{\top}.
$$

Then $z_\ell^{P}(x)=Pz_\ell(x)$ while the MLP output, residual stream, logits, and next-token predictions remain unchanged. Probes must therefore be trained and evaluated on $z_\ell$, captured immediately before the down projection; residual-stream probes are not informative for this intervention.

## Protocol

- **Model:** pinned Mistral-7B-v0.3 at 75% depth, corresponding to block 24 of 32 and an intermediate width of 14,336.
- **Data:** the protected WildGuardMix protocol with training, validation, and test sizes of 12,000, 2,000, and 1,699; data seeds 42 and 137.
- **Probes:** linear, degree-2 CP, and one-hidden-layer MLP probes.
- **Interventions:** identity and exact MLP-neuron permutations seeded by 42 and 137.
- **Comparisons:** reference, identity, naïve transfer, analytic transport, label-free exact feature matching, and inverse-transport control.
- **Function gate:** fail-fast on eight prompts, followed by all 1,699 protected test prompts in FP64. Every intervention must preserve logits within the combined absolute-relative tolerance, retain 100% next-token agreement, and satisfy MLP activation equivariance.

## Outcomes

Primary outcomes are the raw AUROC gap, analytic recovery fraction, and activation-estimated recovery fraction. Coordinate-induced failure requires reference AUROC at least 0.75, an AUROC gap of at least 0.10, and a paired 95% bootstrap interval excluding zero. Analytic and estimated recovery succeed at 95% recovery. Secondary metrics retain AUPRC, accuracy, balanced accuracy, precision, recall, F1, calibration, confusion counts, thresholds, TPR at 1% and 5% FPR, alignment diagnostics, and row-level predictions.

Any failed function gate invalidates probe evaluation. A successful result would show that coordinate-induced transfer failure is not specific to residual representations; it would not establish that natural cross-model MLP features differ only by permutation.
