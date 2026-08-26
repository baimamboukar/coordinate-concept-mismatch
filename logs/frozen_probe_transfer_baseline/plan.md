# Frozen Probe Transfer Baseline

## Objective

Determine whether a probe trained on one modern language model retains predictive performance when applied unchanged to another model with a compatible activation width. The completed Pythia pilot established same-architecture transfer failure, and the completed core phase established Llama–Qwen failure with a Llama–Nemotron lineage control. The pending extension tests whether that result generalizes to Mistral and Granite.

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

## Current scope

- **Pythia pilot — completed:** `EleutherAI/pythia-410m` and `EleutherAI/pythia-410m-seed1`. This same-architecture comparison tests the pipeline and whether a measurable transfer gap exists before expensive extraction.
- **Primary independent-family comparison:** Llama-3.1-8B-Instruct $\rightarrow$ Qwen3-8B and Qwen3-8B $\rightarrow$ Llama-3.1-8B-Instruct.
- **Lineage control:** Llama-3.1-8B-Instruct $\leftrightarrow$ Llama-3.1-Nemotron-Nano-8B-v1.
- **Exploratory comparison:** Qwen3-8B $\leftrightarrow$ Nemotron.
- **Cross-family extension:** add mistralai/Mistral-7B-v0.3 and ibm-granite/granite-3.3-8b-base at immutable revisions. The ten primary directions are every new directed pair among Llama, Qwen, Mistral, and Granite; pairs involving Nemotron remain controls or exploratory.

All five models expose 4,096-dimensional residual streams, permitting direct frozen transfer without an adapter. The Pythia pilot, completed core phase, and cross-family extension answer the same question and remain one experiment.

## Shared protocol

- **Data:** reuse the exact five WildGuardMix splits from the Pythia pilot: 12,000 balanced training rows and 2,000 validation rows under seeds 42 and 137, plus the protected 1,699-row test set. If the prior JSON payload is unavailable, rematerialize it from the pinned dataset and deterministic split procedure, then require exact row-ID and label agreement with the saved Pythia activations. No new split selection is permitted.
- **Activations:** raw prompts, final non-padding token, 25%, 50%, 75%, and 100% model depth; 75% is primary.
- **Probes:** L2 logistic regression is primary. Low-rank degree-2 CP and a width-32 one-hidden-layer MLP test whether shallow nonlinear capacity changes the conclusion.
- **Isolation:** preprocessing, model selection, early stopping, and thresholds use only source train/validation activations. No target labels enter source-probe selection.
- **Evaluation:** report source oracle, frozen transfer, and target oracle on identical test rows for every prespecified direction and both data seeds.

## Metrics and decision rules

Primary outcomes are target AUROC and paired AUROC transfer gap at 75% depth. A directed transfer fails only when source and target-oracle AUROC are each at least 0.75, the gap is at least 0.10, and its paired 95% bootstrap interval excludes zero.

Secondary outcomes are AUPRC, accuracy, balanced accuracy, precision, recall, F1, calibration error, `tn/fp/fn/tp`, TPR at 1% and 5% FPR, achieved target operating points under source thresholds, and results by layer. Row IDs, labels, scores, probabilities, predictions, and thresholds are retained.

For the extension, the comparison-level rule is unchanged. A broad cross-family result requires all twenty primary 75%-depth linear direction-seed comparisons—ten directions under two data seeds—to meet the failure rule. Otherwise, conclusions remain pair-specific. The full five-model baseline must contain exactly 300 metric rows, 509,700 row-level predictions, 240 transfer-gap rows, and ten probe bundles.

## Workflow

1. Preserve and verify the existing five prepared split files, then stage the identical archive on two isolated H100 workers. Run a real checkpoint preflight before launching Mistral and Granite extraction in parallel; each worker extracts all five splits and all four depths.
2. Validate completion metadata, repeatability, shapes, finite values, truncation, and exact row-ID and label agreement on each worker. Upload directly from that worker to Hugging Face, then anonymously re-download and byte-verify there. Activation tensors must not transit through or remain on the coordinating machine.
3. Run the `transfer` stage from `configs/studies/modern_models.yaml` in a clean worker staging root. The pipeline materializes the canonical activation cache, tracks training offline in W&B, and publishes probes and results directly from the worker.
4. Validate all primary, secondary, low-FPR, confusion, calibration, threshold, and row-level outputs before uploading the necessary artifacts to Hugging Face. Upload is deferred on workers.
5. Update the single baseline report and terminate both experiment GPUs after artifact retrieval is verified.

## Interpretation boundary

This experiment establishes whether transfer failure exists and how broadly it occurs. It cannot determine whether a gap is caused by coordinate mismatch or different learned representations; controlled symmetries and alignment address those questions next.
