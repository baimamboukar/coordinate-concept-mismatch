# Coordinate or Concept Mismatch? Disentangling Cross-Model Probe Transfer Failure

## Motivation and research question

Cross-model probe transfer is empirically uneven: a probe trained on one model may transfer successfully, degrade substantially, or recover after representation alignment. This project asks:

> When a probe fails to transfer, how much of the failure is caused by a change of internal coordinates, and how much cannot be explained by known coordinate transformations?

Unlike studies that evaluate alignment only between independently trained models, we first construct functionally identical models whose representational relationship is known exactly. This provides a controlled benchmark for determining whether alignment methods can recover probe transfer when coordinate mismatch is the only source of failure.

## Formal setup

Let $f_{\theta_s}$ and $f_{\theta_t}$ denote source and target models. For an input $x$, their activations at layer $\ell$ are

$$
h_s^\ell(x)\in\mathbb{R}^{d_s},
\qquad
h_t^\ell(x)\in\mathbb{R}^{d_t}.
$$

A probe $p_\phi$ is trained on labelled source activations:

$$
\phi_s^\star
=
\arg\min_\phi
\mathbb{E}_{(x,y)\sim\mathcal{D}_{\mathrm{train}}}
\left[
\mathcal{L}\bigl(p_\phi(h_s^\ell(x)),y\bigr)
\right].
$$

Direct transfer evaluates the same probe on target activations. More generally, an alignment map $A_{t\rightarrow s}$ transforms target activations into source coordinates:

$$
R_{s\rightarrow t}(A)
=
\mathbb{E}_{(x,y)\sim\mathcal{D}_{\mathrm{test}}}
\left[
\mathcal{L}\bigl(
p_{\phi_s^\star}(A_{t\rightarrow s}h_t^\ell(x)),y
\bigr)
\right].
$$

For a function-preserving parameter transformation $g$ from a symmetry group $\mathcal{G}$,

$$
f_{g\theta_s}(x)=f_{\theta_s}(x)
\qquad
\text{for all }x,
$$

while the internal activation may transform as

$$
h_{g\theta_s}^\ell(x)
=
\rho_\ell(g)\,h_{\theta_s}^\ell(x).
$$

Consequently, a linear probe with weight $w_s$ should be transported as

$$
w_g=\rho_\ell(g)^{-T}w_s.
$$

Failure before transport, followed by recovery under the exact inverse transformation, constitutes direct evidence of coordinate-induced probe failure.

## Experimental workflow

The first stage applies algebraically exact transformations to trained open-weight transformers. These will include MLP-neuron permutations, valid positive rescalings, attention-head or component permutations, and architecture-specific normalization symmetries. Only transformations that are exact for the selected architecture will be used. Functional equivalence will be checked through parameter-level identities and numerical agreement of logits across a held-out corpus.

For each layer, token position, probe task, and symmetry type, we compare:

1. probe performance on the original model;
2. naive transfer to the transformed model;
3. analytically transported probe performance;
4. performance after estimating the alignment from activations alone.

The second stage considers independently trained checkpoints. Alignment maps are fitted using paired, unlabelled calibration inputs, with probe-training and evaluation examples kept disjoint. We compare direct transfer, symmetry-constrained canonicalization, permutation and diagonal matching, orthogonal or affine activation matching, quotient-based alignment, and a target-trained probe as an oracle.

For a higher-is-better metric $S$, the recovered fraction of the transfer gap is

$$
\operatorname{Recovery}(A)
=
\frac{
S_{s\rightarrow t}(A)-S_{s\rightarrow t}(I)
}{
S_{t\rightarrow t}-S_{s\rightarrow t}(I)
}.
$$

Results will be reported across seeds with bootstrap confidence intervals and decomposed by layer, task, symmetry, model pair, and probe class. Cross-family experiments will then test how recovery changes when architecture and training history are no longer controlled.

The principal outcome is a quantitative decomposition of probe-transfer failure. Residual failure will be interpreted conservatively: it identifies variation not explained by the tested alignment class, rather than proving that the models encode fundamentally different concepts.
