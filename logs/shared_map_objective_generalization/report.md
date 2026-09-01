# September 1, 2026 | Shared-Map Objective Generalization

[Plan](plan.md) | [SmolLM artifacts](https://huggingface.co/buckets/baimamboukar/coordinate-concept-mismatch/tree/studies/shared-map-objective-generalization/smollm-shared-map-objective-generalization) | [OLMo artifacts](https://huggingface.co/buckets/baimamboukar/coordinate-concept-mismatch/tree/studies/shared-map-objective-generalization/olmo1-shared-map-objective-generalization) | [W&B runs](https://wandb.ai/JinesisLab/coordinate-concept-mismatch/groups/shared_map_objective_generalization)

## Objective and protocol

We tested whether stronger map-fitting objectives produce a task-general alignment or only preserve
the tasks used to fit them. The study used SmolLM-1.7B/SmolLM2-1.7B and independently trained
Ai2/AMD OLMo 1B pairs. Maps were fitted jointly on SST-2 and WildGuard, then evaluated without
refitting on AG News and MNLI. Both directions, seeds 42 and 137, the 75% residual depth, disjoint
calibration/evaluation rows, linear primary probes, and degree-2/MLP secondary probes were fixed in
advance.

Fit compatibility required at least 0.75 median recovery and retention, 3/4 substantial recoveries,
and no substantial shuffled recovery. The inherited held-out rule required 0.50 recovery, 0.75
retention, 3/4 substantial recoveries, and no substantial shuffled recovery.

## Results

Held-out entries are median retention / recovery. “Pass” applies the full held-out rule.

| Pair | Shared-map condition | Worst fit recovery | AG News | MNLI |
| --- | --- | ---: | ---: | ---: |
| SmolLM | uniform fixed | 0.723 | not run | not run |
| SmolLM | reconstruction selected | 0.768 | 0.760 / 0.736 — pass | 0.522 / 0.507 — fail |
| SmolLM | probe selected | 0.848 | 0.796 / 0.772 — pass | 0.542 / 0.527 — fail |
| SmolLM | probe bank | 0.935 | 0.739 / 0.714 — fail | 0.508 / 0.494 — fail |
| OLMo | uniform fixed | 0.810 | 0.467 / 0.459 — fail | 0.314 / 0.299 — fail |
| OLMo | reconstruction selected | 0.798 | 0.439 / 0.431 — fail | 0.291 / 0.277 — fail |
| OLMo | probe selected | 0.864 | 0.458 / 0.449 — fail | 0.253 / 0.242 — fail |
| OLMo | probe bank | 0.956 | 0.443 / 0.435 — fail | 0.288 / 0.274 — fail |

SmolLM therefore shows limited task-dependent generalization to AG News, but not MNLI. OLMo shows
no held-out success. The probe-bank lift reached near-perfect fit-task recovery yet failed every
held-out test, so greater fitted-bank capacity did not produce a task-general map. All 46 W&B runs
finished, and the linked artifacts retain the complete metric contract, diagnostics, probe weights,
and row-level predictions.

## Conclusion and next step

The evidence supports task-conditioned linear recoverability under the tested map class, not a
global canonical coordinate system. It does not establish intrinsic concept mismatch. We should
now freeze this zero-shot result and test prespecified task-specific low-rank corrections on the
same locked partitions.
