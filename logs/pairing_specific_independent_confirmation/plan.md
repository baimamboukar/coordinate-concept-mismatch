# Pairing-Specific Independent Confirmation

[Hugging Face artifacts](https://huggingface.co/buckets/baimamboukar/coordinate-concept-mismatch/tree/studies/pairing-specific-independent-confirmation) | [Weights & Biases](https://wandb.ai/JinesisLab/coordinate-concept-mismatch/groups/pairing_specific_independent_confirmation)

## Objective

Test whether the correspondence-dependent low-rank repair observed on AG News and MNLI replicates
on tasks that were not used to select the adapter, rank, calibration budget, controls, or decision
rule. QNLI and QQP are fixed before activation extraction. A failed task is retained as a failed
endpoint and is never replaced.

## Protocol

The model pairs are SmolLM-1.7B/SmolLM2-1.7B and independently trained Ai2/AMD OLMo 1B. Shared
affine maps remain fitted on SST-2 and WildGuard with the previously selected probe-aware objective.
For each new task, model direction, and seed in $$\{42,137\}$$, the frozen rank-8 adapter uses 256
paired, unlabeled calibration activations at normalized depth 0.75:

$$
\Delta W_p=\underset{\operatorname{rank}(\Delta W)\leq 8}{\arg\min}
\left\|H_t\Delta W-\left(H_s-A_0(H_t)\right)\right\|_F^2
+\lambda\left\|\Delta W\right\|_F^2.
$$

The primary null comprises 99 independently seeded residual-shuffle fits. Ninety-nine
source-shuffle fits and rank-8 CORAL are secondary controls. Calibration, diagnostic, probe-training,
and protected test rows are disjoint. Linear probes remain primary; all previously specified
classification, calibration, low-FPR, confusion, threshold, interval, diagnostic, and row-level
outputs are retained.

## Qualification and decisions

Before fitting task adapters, each task/model-pair endpoint must show all four primary frozen
transfer failures, median same-task affine recovery of at least 0.75, substantial same-task recovery
in at least 3/4 seed-direction contexts, and no substantial shuffled recovery. A non-qualifying
endpoint is skipped computationally but counts as a confirmatory failure.

For a qualified endpoint, success requires median paired recovery at least 0.50, median retention
at least 0.75, median pairing-specific lift at least 0.10, pooled empirical $$p\leq0.05$$, and at
least 3/4 contexts beating every residual shuffle. Empirical probability is

$$
p=\frac{1+\sum_{k=1}^{99}\mathbf{1}[R_{r,k}\geq R_p]}{100}.
$$

Paper-level confirmation requires at least three of four model-pair/task endpoints to pass, with
at least one pass for each task and each model pair. The four pooled probabilities are adjusted by
Holm's procedure at familywise $$\alpha=0.05$$. Outcomes remain confirmatory only under this locked
rule; other probe families or adapter capacities are outside this experiment.
