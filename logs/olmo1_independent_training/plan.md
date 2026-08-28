# OLMo 1 Independent-Training Transfer

## Objective

Test whether probe-transfer failure and coordinate-based recovery persist between architecture-identical models trained independently from scratch. `allenai/OLMo-1B-hf` and `amd/AMD-OLMo-1B` share the 16-layer, 2,048-dimensional OLMo architecture but differ in initialization, training provider, hardware, corpus version, and token budget. This is therefore an independent-training comparison, not a seed-only causal intervention.

## Formal test

For source model $s$, target model $t$, depth $\ell$, and source-trained probe $p_s$, raw transfer is

$$
S_{s\rightarrow t}
=
\operatorname{AUROC}\!\left(p_s(h_t^\ell(x)),y\right).
$$

After fitting an unlabeled target-to-source map $A$ on paired training prompts, recovery is

$$
R(A)
=
\frac{\operatorname{AUROC}(p_s(Ah_t^\ell(x)),y)-S_{s\rightarrow t}}
{S_{t\rightarrow t}-S_{s\rightarrow t}}.
$$

## Protocol

- **Task:** the pinned SST-2 split used in the preceding OLMo 2 study, with 12,000 training, 2,000 validation, and 872 protected test examples under data seeds 42 and 137.
- **Input control:** both checkpoints use the same pinned Ai2 tokenizer with special-token insertion disabled, making the compared prompt representation identical before model execution.
- **Primary baseline:** bidirectional linear-probe transfer at 75% depth. The experiment supports general independent-training failure if at least three of four seed-direction comparisons pass the existing oracle, 0.10 AUROC-gap, and paired-confidence-interval gate, with a median gap of at least 0.10.
- **Primary recovery:** among eligible failures, permutation-plus-positive-diagonal alignment must obtain median recovery of at least 50%; at least half must individually improve by at least 0.05 AUROC, recover at least 50% of the gap, and have an improvement interval above zero.
- **Secondary analyses:** exact permutation, orthogonal, affine, quotient, and shuffled-pair maps; 25%, 50%, and 100% depths; degree-2 and MLP probes at 75% depth.

Primary metrics are AUROC transfer gap, aligned AUROC improvement, recovery fraction, and residual oracle gap. Secondary metrics retain AUPRC, accuracy, balanced accuracy, precision, recall, F1, calibration, confusion counts, source-threshold TPR and achieved FPR at 1% and 5% FPR, thresholds, diagnostics, and row-level predictions.

If the restricted map succeeds, the result extends the OLMo 2 stage-transition finding beyond a shared parent. Flexible-map recovery alone establishes linear recoverability, not parameter symmetry. Failure of all maps would instead support a stronger concept- or feature-organization mismatch.
