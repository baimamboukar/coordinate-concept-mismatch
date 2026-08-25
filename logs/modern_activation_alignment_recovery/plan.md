# Modern Activation-Alignment Recovery

## Objective

Estimate how much of the natural frozen-probe transfer gap between modern models can be recovered by a map fitted only from paired, unlabeled activations. The completed baseline supplies the fixed probes, raw transfer scores, target oracles, thresholds, and protected test set. This experiment changes only the target-to-source activation coordinates.

## Formal setup

For source model $s$, target model $t$, normalized depth $\ell$, and frozen source probe $p_s$, fit $A_{t\rightarrow s}$ on paired training prompts without using safety labels. Evaluate

$$
S_{s\rightarrow t}(A)
=
\operatorname{AUROC}\!\left(
p_s\!\left(A_{t\rightarrow s}h_t^\ell(x)\right),y
\right).
$$

Let $S_{s\rightarrow t}(I)$ be raw transfer and $S_{t\rightarrow t}$ the target-trained oracle from the baseline. When the raw gap is at least 0.10, define

$$
R(A)
=
\frac{S_{s\rightarrow t}(A)-S_{s\rightarrow t}(I)}
{S_{t\rightarrow t}-S_{s\rightarrow t}(I)}.
$$

The denominator is not interpreted when the raw gap is below 0.10.

## Prespecified scope

- **Primary pair:** Llama $\leftrightarrow$ Qwen.
- **Lineage control:** Llama $\leftrightarrow$ Nemotron. Its baseline AUROC gaps are small, so it tests non-degradation and threshold stability rather than recovery fraction.
- **Exploratory pair:** Qwen $\leftrightarrow$ Nemotron.
- **Primary representation:** final prompt token at 75% normalized depth.
- **Primary probe:** frozen linear probe.
- **Secondary probes:** frozen degree-2 CP and one-hidden-layer MLP probes.
- **Repetitions:** data seeds 42 and 137 are separate alignment-fitting repetitions on the same protected 1,699-row test set, not independent test replications.

All models have 4,096-dimensional residual streams. The experiment reuses the exact 12,000 training, 2,000 validation, and 1,699 test rows from the completed baseline.
Execution requires one H100 on a host advertising CUDA 13.0 or newer driver support, matching the frozen PyTorch environment.

## Alignment methods

The primary restricted map is feature permutation followed by a positive diagonal affine transformation. Strict permutation is a more restrictive comparison. Orthogonal Procrustes and full affine Ridge are flexible linear-recoverability bounds, not parameter-symmetry estimates. Quotient Ridge is evaluated only for linear probes. Affine Ridge fitted after shuffling source-target prompt pairs is the negative control.

Every map is shared across probe families, fitted on paired training activations, and diagnosed on paired validation activations. Test activations and all labels are excluded from fitting and selection.

## Outcomes and decision rules

Primary outcomes are aligned AUROC improvement, recovery fraction, and residual oracle gap for permutation-diagonal alignment on the 75%-depth linear Llama $\leftrightarrow$ Qwen comparisons.

A direction-seed comparison counts as substantial restricted recovery only when:

- source and target-oracle AUROC are each at least 0.75;
- the fixed raw transfer gap is at least 0.10;
- aligned AUROC improves by at least 0.05;
- the paired 95% bootstrap interval for improvement excludes zero; and
- recovery fraction is at least 0.50.

The experiment-level restricted-coordinate claim requires all four primary comparisons—both directions under both fitting seeds—to satisfy this rule. Otherwise, results remain direction-specific.

Flexible-map recovery is secondary and cannot establish a parameter symmetry. Recovery by the shuffled-pair control weakens any interpretation that depends on prompt correspondence. For the lineage control, report aligned-minus-raw AUROC and the change in deviation from nominal 1% and 5% FPR; do not headline recovery fractions with small denominators.

## Metrics and retained evidence

Primary metrics are aligned AUROC improvement, recovery fraction, and residual oracle gap. Secondary metrics are AUROC, AUPRC, accuracy, balanced accuracy, precision, recall, F1, expected calibration error, $tn/fp/fn/tp$, TPR at 1% and 5% FPR, achieved target FPR and TPR under source thresholds, alignment relative RMSE, alignment cosine similarity, and negative-control results.

Every condition retains row ID, label, score, probability, prediction, balanced-accuracy threshold, and source operating thresholds. Paired 95% bootstrap intervals use 2,000 resamples.

The prespecified primary-depth run must produce 264 metric rows, 448,536 prediction rows, 192 recovery rows, and 72 alignment-diagnostic rows.

## Workflow

1. Download the verified modern baseline prefix from Hugging Face and reproduce every raw, source-oracle, and target-oracle AUROC before fitting maps.
2. Fit each alignment on paired training activations and evaluate reconstruction diagnostics on validation activations.
3. Evaluate frozen probes on the untouched test rows, compute paired recovery intervals, and validate all expected output counts and metric fields.
4. Track the run offline in Weights & Biases, retrieve it to the trusted machine, and upload the verified results to the project Hugging Face bucket.
5. Write the concise dated report and terminate the compute worker only after retrieval and remote artifact verification.

## Interpretation boundary

Restricted recovery estimates the component compatible with that tested activation-coordinate class. Flexible linear recovery shows linear predictability only. Failure of the tested maps does not prove concept mismatch, and success does not show that independently trained models are related by an exact parameter-space symmetry. Cross-family generality remains incomplete until Mistral is added.
