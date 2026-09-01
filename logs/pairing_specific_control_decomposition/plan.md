# Pairing-Specific Control Decomposition

## Objective

Determine whether the gains from task-specific low-rank correction require exact source-target
activation correspondence, or can be explained by task-level activation distributions. This is a
diagnostic follow-up on previously evaluated tasks, not an independent confirmation.

## Formal setup

Let the frozen shared map be $$A_0(h)=hW_0+b_0$$. Using 256 unlabeled calibration pairs, the paired
rank-8 adapter fits the residual

$$
R=H_s-A_0(H_t), \qquad
\Delta W_p=\underset{\operatorname{rank}(\Delta W)\leq 8}{\arg\min}
\left\|H_t\Delta W-R\right\|_F^2+\lambda\left\|\Delta W\right\|_F^2.
$$

Three controls isolate alternative mechanisms:

1. Twenty residual-shuffle fits replace $$R$$ with $$P_kR$$. They preserve the residual marginal
   and adapter class while removing target-residual correspondence; this is the primary null.
2. Twenty source-shuffle fits use $$P_kH_s-A_0(H_t)$$, retaining only unpaired source and target
   marginals under the same rank and sample budget.
3. A rank-8 CORAL adapter aligns the first two moments of $$A_0(H_t)$$ to $$H_s$$ without using row
   correspondence. This is a secondary distribution-matching baseline.

For each seed and direction, let $$R_p$$ be paired recovery and $$R_{r,k}$$ residual-shuffle
recovery. Pairing-specific lift is

$$
D=R_p-\operatorname{median}_k R_{r,k},
$$

with empirical one-sided probability

$$
p=\frac{1+\sum_k\mathbf{1}[R_{r,k}\geq R_p]}{21}.
$$

## Protocol and decision rule

- Model pairs: SmolLM-1.7B/SmolLM2-1.7B and independently trained Ai2/AMD OLMo 1B.
- Tasks: AG News and MNLI; probe-selected shared maps only.
- Frozen linear probe at normalized depth 0.75; seeds 42 and 137; both directions.
- Calibration, calibration-validation, and protected test partitions remain disjoint.
- The endpoint supports pairing-specific repair when median paired recovery is at least 0.50,
  median retention is at least 0.75, median $$D$$ is at least 0.10, pooled $$p\leq0.05$$, and at
  least three of four seed-direction comparisons beat every residual-shuffle repeat.

Primary outputs are recovery, retention, pairing-specific lift, empirical probability, and control
wins. Secondary outputs retain CORAL/source-shuffle recovery, AUROC, AUPRC, accuracy, balanced
accuracy, precision, recall, F1, calibration, confusion counts, thresholds, TPR at 1% FPR,
bootstrap intervals, diagnostics, and row-level predictions. Compute runs from YAML; inputs and
outputs move directly between the worker and the public Hugging Face bucket, with W&B telemetry.
