# Pythia Checkpoint-Alignment Recovery

## Objective

The baseline found a large probe-transfer gap between independently trained Pythia-410M checkpoints, and the controlled permutation experiment showed that coordinates alone can cause such a gap. This experiment now estimates how much of the *natural* Pythia gap can be recovered by alignment learned from paired, unlabelled activations.

## Formal setup

For source checkpoint $s$, target checkpoint $t$, and layer $\ell$, an alignment map $A_{t\rightarrow s}$ is fitted on the same training prompts processed by both models, without using their labels. The frozen source probe is then evaluated on the protected target test activations:

$$
S_{s\rightarrow t}(A)
=
\operatorname{AUROC}\!\left(
p_s\!\left(A_{t\rightarrow s}h_t^\ell(x)\right),y
\right).
$$

With raw transfer score $S_{s\rightarrow t}(I)$ and target-trained oracle $S_{t\rightarrow t}$, the recovered fraction is

$$
R(A)
=
\frac{S_{s\rightarrow t}(A)-S_{s\rightarrow t}(I)}
{S_{t\rightarrow t}-S_{s\rightarrow t}(I)}.
$$

The denominator is fixed by the completed baseline. Target labels are used only after alignment for evaluation.

## Design

- **Models and materials:** reuse the pinned `pythia-410m` and `pythia-410m-seed1` activations, probes, thresholds, 12,000 training pairs, 2,000 validation pairs, and protected 1,699-row test set from the baseline.
- **Directions and repetitions:** evaluate both checkpoint directions under data seeds 42 and 137.
- **Probes:** linear, degree-2 CP, and one-hidden-layer MLP probes at block 18 are primary; linear probes at blocks 6, 12, and 24 are secondary.
- **Isolation:** every map is shared across probe families at a layer and is fitted only from paired training activations. Validation activations diagnose map generalization; test activations and labels never fit or select a map.

We compare raw transfer, target oracle, strict permutation matching, permutation plus positive diagonal affine matching, centered orthogonal Procrustes, and full affine Ridge. The permutation-diagonal map is the primary restricted coordinate model. Orthogonal and affine maps are increasingly permissive linear-recoverability bounds, not parameter-symmetry estimates. For linear probes, quotient Ridge follows the probe-visible SVD construction of [Deep Minds and Shallow Probes](https://arxiv.org/abs/2605.11448), using a fixed $10^{-3}$ relative singular-value threshold and $10^{-4}$ relative Ridge penalty. Affine Ridge fitted after shuffling source-target prompt pairs is the negative control.

## Outcomes and decision rules

Primary outcomes at block 18 are aligned AUROC improvement, recovery fraction, and residual oracle gap, each reported by direction, data seed, and probe family. A comparison counts as substantial recovery only when the original source and target oracles exceed 0.75, the raw gap is at least 0.10, aligned improvement is at least 0.05 with a paired 95% bootstrap interval above zero, and $R(A)\geq0.50$.

Secondary outcomes are AUPRC, accuracy, balanced accuracy, precision, recall, F1, calibration error, `tn/fp/fn/tp`, TPR at 1% and 5% FPR, achieved target operating points under source thresholds, held-out alignment error, and layerwise results. Row IDs, labels, scores, probabilities, predictions, and thresholds are retained.

## Workflow and interpretation boundary

1. Reproduce every raw and oracle baseline score from the frozen artifacts.
2. Fit each alignment on paired training activations and evaluate its validation reconstruction diagnostics.
3. Evaluate frozen probes on the untouched target test set and bootstrap paired recovery statistics.
4. Track the run in W&B, upload verified result artifacts to Hugging Face, and write a concise report.

Recovery by the restricted map estimates the component explained by that tested coordinate class. Additional recovery by orthogonal, affine, or quotient methods establishes broader linear predictability, not a known parameter symmetry. Residual failure means only that the tested alignment classes did not explain the gap; it does not prove concept mismatch.
