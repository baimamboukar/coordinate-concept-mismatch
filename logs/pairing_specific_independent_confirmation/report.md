# September 1, 2026 | Pairing-Specific Independent Confirmation

[Plan](plan.md) | [SmolLM artifacts](https://huggingface.co/buckets/baimamboukar/coordinate-concept-mismatch/tree/studies/pairing-specific-independent-confirmation/smollm-pairing-specific-independent-confirmation) | [OLMo artifacts](https://huggingface.co/buckets/baimamboukar/coordinate-concept-mismatch/tree/studies/pairing-specific-independent-confirmation/olmo1-pairing-specific-independent-confirmation) | [W&B runs](https://wandb.ai/JinesisLab/coordinate-concept-mismatch/groups/pairing_specific_independent_confirmation)

## Objective and protocol

We independently tested the previously selected rank-8, 256-pair correction on QNLI and QQP,
which were fixed before extraction and were not used to choose the map objective, rank, calibration
budget, controls, or decision rule. The model pairs were SmolLM-1.7B/SmolLM2-1.7B and Ai2/AMD
OLMo 1B. Each endpoint used both model directions and seeds 42 and 137, with 99 residual-shuffle
nulls, 99 source-shuffle controls, and rank-8 CORAL.

All four endpoints qualified: frozen transfer failed in 16/16 seed-direction contexts, median
same-task affine recovery ranged from 0.964 to 0.989, same-task recovery was substantial in 16/16
contexts, and shuffled same-task recovery was substantial in 0/16.

## Results

| Pair | Task | Recovery / retention | Pairing lift | Wins | Raw / Holm $$p$$ | Decision |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| SmolLM | QNLI | 0.346 / 0.364 | 0.268 | 4/4 | 0.01 / 0.04 | fail: recovery, retention |
| SmolLM | QQP | 0.388 / 0.395 | 0.386 | 4/4 | 0.01 / 0.04 | fail: recovery, retention |
| OLMo | QNLI | 0.217 / 0.222 | 0.139 | 2/4 | 0.01 / 0.04 | fail: recovery, retention, wins |
| OLMo | QQP | 0.241 / 0.244 | 0.156 | 3/4 | 0.01 / 0.04 | fail: recovery, retention |

The paired correction exceeded the pooled shuffle null at every endpoint, including after Holm
control, so correspondence dependence replicated on both independent tasks and model pairs.
However, no endpoint reached the locked recovery threshold of 0.50 or retention threshold of 0.75;
the paper-level criterion therefore failed with 0/4 endpoint passes.

Across the 16 primary contexts, secondary medians were AUROC 0.661, AUPRC 0.629, accuracy and
balanced accuracy 0.525, precision 0.622, recall 0.260, F1 0.367, calibration error 0.329, and TPR
0.020/0.120 at 1%/5% FPR. The public artifacts retain thresholds, confusion counts, operating-point
metrics, diagnostics, intervals, and row-level predictions.

## Interpretation and next step

Exact activation correspondence carries a reproducible signal, but this fixed low-rank repair is
far too small to explain most transfer failure. The result supports task-conditioned coordinate
repair and leaves a large unexplained transfer gap; it does not support a task-general canonical
coordinate system. The next discriminating experiment should measure rank and calibration-sample
dose response under the same held-out controls, separating insufficient repair capacity from
genuinely non-affine or concept-level mismatch.
