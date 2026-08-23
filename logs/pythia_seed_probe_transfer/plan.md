# Pythia Seed Probe Transfer

## Question

Does a probe trained on one Pythia-410M training run lose performance when applied unchanged to an independently trained checkpoint with the same architecture and training budget?

This is a controlled pilot, not evidence that probes generally fail across modern model families. It tests whether the transfer gap is measurable before we spend substantially more compute.

## Setup

- Checkpoints: original `EleutherAI/pythia-410m` (training seed 1234) and `EleutherAI/pythia-410m-seed1`, each pinned to an exact Hugging Face revision.
- Data: cleaned WildGuardMix prompt-harm labels; 12,000 balanced train rows and 2,000 balanced validation rows under each of sampling seeds 42 and 137; one shared protected test set of 1,699 rows.
- Activations: raw prompt, final non-padding token, blocks 6, 12, 18, and 24; block 18 is primary.
- Probes: L2 logistic regression at all depths; low-rank degree-2 CP and width-32 one-hidden-layer MLP at the primary depth.
- Selection: all preprocessing, hyperparameters, early stopping, and operating thresholds use only source train/validation activations. No target labels enter source-probe selection.

For source checkpoint $s$, target checkpoint $t$, and probe family $a$, the primary transfer gap is

$$
G_{s\rightarrow t}^{a}
=
\operatorname{AUROC}(p_t^a(h_t))
-
\operatorname{AUROC}(p_s^a(h_t)).
$$

The first term is the target-trained oracle; the second is the unchanged source probe evaluated in target coordinates.

## Metrics and decision rule

Primary metrics are target AUROC and the paired AUROC transfer gap at 75% depth. Secondary metrics are AUPRC, accuracy, balanced accuracy, precision, recall, F1, expected calibration error, `tn/fp/fn/tp`, TPR at 1% and 5% FPR, and the achieved target operating points under source-selected thresholds. Every result retains row ID, label, score, probability, prediction, and thresholds.

A directed transfer is called failed only if source and target-oracle AUROC are each at least 0.75, the gap is at least 0.10, and its paired 95% bootstrap interval excludes zero. We report both sampling seeds and both directions regardless of outcome.

## Workflow

1. Materialize and checksum the five prepared split files outside the repository.
2. Extract both checkpoints on the same rows and verify eight-row repeatability, shapes, finiteness, truncation, and cross-checkpoint row alignment.
3. Fit and select probes using source data only; track training in W&B offline on the worker, then sync from the trusted local machine.
4. Evaluate source-oracle, zero-shot transfer, and target-oracle conditions with frozen thresholds.
5. Retrieve and checksum activations, compact probe bundles, metrics, transfer gaps, and row-level predictions; upload them to the project Hugging Face bucket.
6. Destroy the rented GPU and verify its absence from the provider inventory before reporting completion.
