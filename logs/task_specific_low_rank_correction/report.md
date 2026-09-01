# September 1, 2026 | Task-Specific Low-Rank Correction

[Plan](plan.md) | [SmolLM artifacts](https://huggingface.co/buckets/baimamboukar/coordinate-concept-mismatch/tree/studies/task-specific-low-rank-correction/smollm-task-specific-low-rank-correction) | [OLMo artifacts](https://huggingface.co/buckets/baimamboukar/coordinate-concept-mismatch/tree/studies/task-specific-low-rank-correction/olmo1-task-specific-low-rank-correction) | [SmolLM W&B](https://wandb.ai/JinesisLab/coordinate-concept-mismatch/groups/smollm_task_specific_low_rank_correction) | [OLMo W&B](https://wandb.ai/JinesisLab/coordinate-concept-mismatch/groups/olmo1_task_specific_low_rank_correction)

## Objective and protocol

We tested whether the held-out failure of a task-independent cross-model map can be repaired by a
small task-conditioned deviation. For each frozen probe-selected shared map, we fitted a rank-8
correction from 256 paired, unlabeled calibration activations on AG News or MNLI. The shared bias
and probes remained frozen. SmolLM-1.7B/SmolLM2-1.7B and independently trained Ai2/AMD OLMo 1B
pairs were evaluated in both directions with seeds 42 and 137. The locked rule required at least
0.50 median recovery, 0.75 retention of the same-task affine improvement, substantial recovery in
at least 3/4 comparisons, and none under shuffled pairing. Reconstruction-selected maps and the
full rank/data sweep were secondary analyses.

## Results

The table reports the primary probe-selected condition. “Shared” is the frozen task-independent
map; “low-rank” is the rank-8/256 correction.

| Pair | Task | Shared recovery | Low-rank recovery / retention | Substantial | Shuffled substantial | Decision |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| SmolLM | AG News | 0.772 | 0.836 / 0.861 | 4/4 | 3/4 | fail: control |
| SmolLM | MNLI | 0.527 | 0.617 / 0.632 | 3/4 | 0/4 | fail: retention |
| OLMo | AG News | 0.449 | 0.718 / 0.728 | 4/4 | 0/4 | fail: retention |
| OLMo | MNLI | 0.242 | 0.354 / 0.372 | 1/4 | 0/4 | fail: recovery and retention |

The correction improved median recovery for every model-task endpoint, with the largest gain on
OLMo/AG News (+0.269). Nevertheless, no endpoint passed the complete confirmatory rule. SmolLM/AG
News cannot be interpreted as pairing-specific coordinate recovery because its shuffled correction
also recovered 0.628 median gap and was substantial in 3/4 comparisons. OLMo/AG News is the cleanest
positive result, but missed the retention threshold by 0.022. Exploratory capacity scaling raised
OLMo/MNLI recovery to 0.651 at rank 32 with 4,096 pairs, while SmolLM/MNLI remained below 0.70.

## Conclusion and next step

Residual transfer failure is structured and partly task-adaptable, but it is not generally explained
by a small pairing-specific coordinate correction. The next experiment should separate paired
correspondence from task-distribution adaptation using repeated shuffle controls and an explicit
unpaired moment-matching baseline before increasing adapter capacity.
