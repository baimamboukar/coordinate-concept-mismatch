# OLMo 1 Independent-Training Safety Replication

## Objective

Test whether the independent-training result observed on SST-2 generalizes to a safety-relevant concept. The Ai2 and AMD OLMo 1B checkpoints, shared tokenizer, representation sites, probe families, alignment maps, data seeds, and decision rules remain fixed; only the task changes to WildGuardMix prompt harmfulness.

## Formal comparison

For source model $s$, target model $t$, depth $\ell$, and source-trained probe $p_s$, raw transfer is

$$
S_{s\rightarrow t}
=
\operatorname{AUROC}\!\left(p_s(h_t^\ell(x)),y\right).
$$

For an unlabeled target-to-source activation map $A$, recovery relative to the target-trained oracle is

$$
R(A)
=
\frac{\operatorname{AUROC}(p_s(Ah_t^\ell(x)),y)-S_{s\rightarrow t}}
{S_{t\rightarrow t}-S_{s\rightarrow t}}.
$$

## Protocol

- **Data:** pinned WildGuardMix with 12,000 balanced training examples, 2,000 validation examples, and 1,699 protected test examples under seeds 42 and 137; sampling is stratified by adversarial status.
- **Models:** the same pinned, architecture-identical `allenai/OLMo-1B-hf` and `amd/AMD-OLMo-1B` checkpoints used for SST-2. Both receive the same pinned Ai2 tokenizer with automatic special-token insertion disabled.
- **Representations:** final non-padding token at 25%, 50%, 75%, and 100% depth; 75% is primary.
- **Probes and maps:** linear probes are primary; degree-2 CP and one-layer MLP probes are secondary. The alignment ladder is permutation, permutation plus positive diagonal, orthogonal Procrustes, affine Ridge, quotient Ridge, and shuffled-pair affine control.

The baseline generalizes if at least three of four 75%-depth linear comparisons meet the existing oracle, 0.10 AUROC-gap, and paired-confidence-interval criteria, with median gap at least 0.10. Eligible failures retain the original restricted-recovery rule: median recovery at least 50%, with at least half individually improving by 0.05, recovering 50%, and having an improvement interval above zero.

The SST-2 pattern is replicated if the baseline passes, restricted recovery remains below its rule, orthogonal or affine alignment obtains at least 90% median recovery with substantial recovery in 4/4 comparisons, and shuffled alignment recovers none. If the baseline does not fail, the result establishes task dependence and alignment recovery is not interpreted.

Primary metrics are AUROC transfer gap, recovery fraction, aligned improvement, and residual oracle gap. Secondary outputs retain AUPRC, calibration, accuracy, balanced accuracy, precision, recall, F1, confusion counts, thresholds, TPR and achieved FPR at 1% and 5% FPR, diagnostics, and row-level predictions. Heavy artifacts publish directly from the GPU worker to Hugging Face; training runs synchronize to W&B.
