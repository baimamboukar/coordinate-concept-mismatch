# OLMo 1 Cross-Task Alignment-Map Transport

## Objective

Determine whether the dense Ai2↔AMD OLMo alignment is a task-independent change of basis or a task-local match between activation distributions. Maps fitted on unlabeled SST-2 prompts will be evaluated on WildGuardMix probes and activations without refitting; the reverse WildGuardMix→SST-2 transport is evaluated symmetrically.

## Formal test

Let $A^{a}_{s\rightarrow t}$ be a target-to-source map fitted on paired training activations from task $a$. For an evaluation task $b$, cross-task recovery is

$$
R_{a\rightarrow b}
=
\frac{\operatorname{AUROC}\!\left(p_s^b(A^{a}_{s\rightarrow t}h_t^b(x)),y\right)-S_{s\rightarrow t}^b}
{S_{t\rightarrow t}^b-S_{s\rightarrow t}^b}.
$$

Relative to the previously frozen same-task map $A^{b}_{s\rightarrow t}$, improvement retention is

$$
T_{a\rightarrow b}
=
\frac{\operatorname{AUROC}\!\left(p_s^b(A^{a}_{s\rightarrow t}h_t^b(x)),y\right)-S_{s\rightarrow t}^b}
{\operatorname{AUROC}\!\left(p_s^b(A^{b}_{s\rightarrow t}h_t^b(x)),y\right)-S_{s\rightarrow t}^b}.
$$

## Protocol

- **Fixed materials:** the pinned Ai2 and AMD OLMo 1B checkpoints, shared tokenizer, four residual-stream depths, data seeds 42 and 137, and the completed SST-2 and WildGuardMix activation/probe artifacts.
- **Separation:** each map uses only 12,000 paired, unlabeled training prompts from the fit task. No evaluation-task example, label, validation statistic, or probe weight is used to fit the ambient map.
- **Primary analysis:** affine Ridge at 75% depth with linear probes, both model directions, both seeds, and both task-transport directions.
- **Secondary analyses:** permutation, permutation-plus-positive-diagonal, orthogonal Procrustes, quotient Ridge, and shuffled-pair affine control; all depths; degree-2 and MLP probes at 75% depth. Quotient results remain exploratory because their basis depends on evaluation-task probes.

The task-general basis hypothesis is supported if, for both SST-2→WildGuardMix and WildGuardMix→SST-2, affine maps recover at least 50% of the median gap, achieve substantial recovery in at least 3/4 comparisons, retain at least 75% of the median same-task improvement, and the shuffled control is substantial in 0/4. Recovery of at least 75% with 4/4 substantial comparisons in both task directions constitutes strong support. Failure in one task direction indicates distribution-dependent alignment rather than a single demonstrated global map.

Primary outputs are cross-task recovery, improvement retention, AUROC improvement, and residual oracle gap. Secondary outputs retain the full metric, calibration, low-FPR, diagnostic, threshold, confusion-count, and row-level prediction contract. All inputs are read from Hugging Face on the worker; only result tables are published. No language-model inference or local activation storage is required.
