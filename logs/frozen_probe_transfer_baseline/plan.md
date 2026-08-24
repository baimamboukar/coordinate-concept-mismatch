# Frozen Probe Transfer Baseline

## Objective

Determine whether a probe trained on one model retains predictive performance when applied unchanged to another compatible model. This is one experiment executed in stages: a low-cost Pythia pilot establishes feasibility and signal, while the later modern-model phase tests the broad cross-family claim.

## Formal setup

For prompt $x$, model $m$, normalized depth $\ell$, and probe family $a$, let $h_m^\ell(x)$ be the final-prompt-token activation and $p_s^a$ a probe trained only on source-model data. The transfer gap is

$$
G_{s\rightarrow t}^{a}
=
\operatorname{AUROC}(p_t^a(h_t))
-
\operatorname{AUROC}(p_s^a(h_t)),
$$

where the first term is the target-trained oracle and the second is the unchanged source probe evaluated on the target model.

## Model stages

- **Pythia pilot — completed:** `EleutherAI/pythia-410m` and `EleutherAI/pythia-410m-seed1`. This same-architecture comparison tests the pipeline and whether a measurable transfer gap exists before expensive extraction.
- **Modern-model phase — pending:** Llama-3.1-8B, Mistral-7B-v0.3, and Qwen3-8B form the independent-family matrix. Llama-3.1-Nemotron-Nano-8B is a lineage control. Their 4,096-dimensional residual streams permit direct frozen transfer without an adapter.

The Pythia pilot and modern-model phase answer the same question with the same protocol; they are not separate experiments.

## Shared protocol

- **Data:** cleaned WildGuardMix harmful-versus-unharmful prompts; 12,000 balanced training rows and 2,000 validation rows under seeds 42 and 137, plus one protected 1,699-row test set.
- **Activations:** raw prompts, final non-padding token, 25%, 50%, 75%, and 100% model depth; 75% is primary.
- **Probes:** L2 logistic regression is primary. Low-rank degree-2 CP and a width-32 one-hidden-layer MLP test whether shallow nonlinear capacity changes the conclusion.
- **Isolation:** preprocessing, model selection, early stopping, and thresholds use only source train/validation activations. No target labels enter source-probe selection.
- **Evaluation:** report source oracle, frozen transfer, and target oracle on identical test rows in both directions and for both data seeds.

## Metrics and decision rules

Primary outcomes are target AUROC and paired AUROC transfer gap at 75% depth. A directed transfer fails only when source and target-oracle AUROC are each at least 0.75, the gap is at least 0.10, and its paired 95% bootstrap interval excludes zero.

Secondary outcomes are AUPRC, accuracy, balanced accuracy, precision, recall, F1, calibration error, `tn/fp/fn/tp`, TPR at 1% and 5% FPR, achieved target operating points under source thresholds, and results by layer. Row IDs, labels, scores, probabilities, predictions, and thresholds are retained.

The Pythia pilot supports only a same-architecture result. Broad linear-probe transfer failure requires at least four of the six directed Llama–Mistral–Qwen transfers to fail with a median gap of at least 0.10. Nemotron and nonlinear probes are reported separately.

## Workflow

1. Use the completed Pythia pilot to validate extraction, probe training, evaluation, and artifact handling.
2. Run the unchanged protocol on the modern-model matrix when compute is available.
3. Evaluate every source–target direction with paired bootstrap intervals and the full metric set.
4. Track training in W&B, store derived artifacts in Hugging Face, and update one concise baseline report.

## Interpretation boundary

This experiment establishes whether transfer failure exists and how broadly it occurs. It cannot determine whether a gap is caused by coordinate mismatch or different learned representations; controlled symmetries and alignment address those questions next.
