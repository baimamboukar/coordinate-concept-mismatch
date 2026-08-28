# OLMo 1 Pooled-Map Generalization to Held-Out BoolQ

## Objective

Determine whether the failed SST-2↔WildGuard map transport reflects inadequate activation-distribution coverage or genuinely task-conditional alignment. Affine Ai2↔AMD OLMo maps will be fitted on SST-2 alone, WildGuardMix alone, or an equally weighted pool, then evaluated without refitting on held-out BoolQ probes and activations.

## Formal test

For fit distribution $F \in \{S,W,P\}$ and BoolQ evaluation distribution $B$, recovery is

$$
R_F^B
=
\frac{\operatorname{AUROC}\!\left(p_s^B(A^F_{s\rightarrow t}h_t^B),y^B\right)-S_{s\rightarrow t}^B}
{S_{t\rightarrow t}^B-S_{s\rightarrow t}^B}.
$$

Improvement retention relative to a same-task BoolQ map $A^B_{s\rightarrow t}$ is

$$
T_F^B
=
\frac{\operatorname{AUROC}\!\left(p_s^B(A^F_{s\rightarrow t}h_t^B),y^B\right)-S_{s\rightarrow t}^B}
{\operatorname{AUROC}\!\left(p_s^B(A^B_{s\rightarrow t}h_t^B),y^B\right)-S_{s\rightarrow t}^B}.
$$

## Protocol

- **Held-out task:** pinned `google/boolq`, formatted as `Question: ...\nPassage: ...`; answer truth is the binary label. The official 3,270-row validation split is protected for testing. Per seed, 6,000 balanced training and 1,000 balanced validation examples are drawn from the training split.
- **Equal fit budgets:** SST-2-only and WildGuard-only maps each use 6,000 paired training examples. The pooled map uses 3,000 from each task. Labels are never used for map fitting.
- **Fixed factors:** pinned Ai2 and AMD OLMo 1B checkpoints, shared tokenizer, seeds 42 and 137, 75% residual-stream depth, both model directions, and the existing probe and metric protocol.
- **Primary analysis:** affine Ridge with linear probes. Orthogonal Procrustes and shuffled-pair affine alignment are comparators; degree-2 and MLP probes are secondary.

BoolQ qualifies as a transport test only if all 4/4 primary frozen transfers meet the existing failure rule and its same-task affine map recovers at least 75% of the median gap with substantial recovery in at least 3/4 comparisons. If this gate fails, BoolQ will be reported as non-qualifying and no cross-task result will be interpreted; another task will not be substituted post hoc.

The task-general pooled-map criterion passes if median pooled recovery is at least 50%, at least 3/4 comparisons show substantial recovery, median same-task improvement retention is at least 75%, and the shuffled control is substantial in 0/4. A coverage advantage additionally requires pooled median recovery to exceed both equal-budget single-task maps by at least 10 percentage points and to outperform each in at least 3/4 paired comparisons. Recovery of at least 75% with 4/4 substantial comparisons constitutes strong support.

Primary outputs are recovery, improvement retention, AUROC improvement, residual oracle gap, and paired pooled-versus-single differences. Secondary outputs preserve AUROC, AUPRC, calibration, low-FPR operating points, thresholds, confusion counts, diagnostics, and row-level predictions. Activations and results move directly between the worker and Hugging Face; only concise reports remain in Git.
