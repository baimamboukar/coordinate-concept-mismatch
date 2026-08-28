# OLMo 2 Stage-Transition Alignment

## Objective

Determine whether the SST-2 probe-transfer failures observed between the shared Stage-1 parent and three Stage-2 OLMo 2 1B checkpoints are explained by a restricted coordinate mismatch. This is a prospective mechanistic follow-up selected after observing the transfer baseline; it is not an independent replication.

## Formal test

For source checkpoint $s$, target checkpoint $t$, layer $\ell$, and frozen source probe $p_s$, fit an unlabeled target-to-source map $A$ on paired training prompts and evaluate

$$
S_{s\rightarrow t}(A)
=
\operatorname{AUROC}\!\left(p_s(Ah_t^\ell(x)),y\right).
$$

Relative to raw transfer and the target-trained oracle, recovery is

$$
R(A)
=
\frac{S_{s\rightarrow t}(A)-S_{s\rightarrow t}(I)}
{S_{t\rightarrow t}-S_{s\rightarrow t}(I)}.
$$

Maps use no labels, validation examples diagnose generalization, and protected test examples are used only for final evaluation.

## Prespecified protocol

- **Comparisons:** all six directed Stage-1↔Stage-2 pairs under data seeds 42 and 137. No pair is selected by its baseline outcome.
- **Primary analysis:** linear probes at 75% depth and permutation-plus-positive-diagonal alignment, restricted to comparisons satisfying the existing oracle, AUROC-gap, and paired-confidence-interval failure gate.
- **Primary decision:** meaningful restricted recovery requires median recovery of at least 50%, with at least half of eligible comparisons individually improving by at least 0.05 AUROC, recovering at least 50% of their gap, and having an improvement interval above zero.
- **Secondary analyses:** 25%, 50%, and 100% depths; exact permutation; degree-2 and MLP probes at 75%; orthogonal, affine, and quotient maps. These are descriptive and do not replace the primary result.
- **Negative control:** shuffled-pair affine alignment must not satisfy the substantial-recovery rule.

Primary outcomes are aligned AUROC improvement, recovery fraction, and residual oracle gap. Secondary outputs retain AUPRC, accuracy, balanced accuracy, precision, recall, F1, calibration, confusion counts, TPR at 1% and 5% FPR, achieved FPR, thresholds, alignment diagnostics, and row-level predictions.

The completed SST-2 activations, probes, and baseline metrics are reused without extraction or retraining. Inputs materialize directly from Hugging Face on one labeled worker; results publish back to Hugging Face and W&B without transiting through the coordinating machine.
