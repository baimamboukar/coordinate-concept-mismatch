# September 1, 2026 | Pairing-Specific Control Decomposition

[Plan](plan.md) | [SmolLM artifacts](https://huggingface.co/buckets/baimamboukar/coordinate-concept-mismatch/tree/studies/pairing-specific-control-decomposition/smollm-pairing-specific-control-decomposition) | [OLMo artifacts](https://huggingface.co/buckets/baimamboukar/coordinate-concept-mismatch/tree/studies/pairing-specific-control-decomposition/olmo1-pairing-specific-control-decomposition) | [W&B runs](https://wandb.ai/JinesisLab/coordinate-concept-mismatch/groups/pairing_specific_control_decomposition)

## Objective and protocol

We tested whether the gains from the preceding task-specific low-rank correction require exact
source-target activation correspondence. For each frozen probe-selected shared map, we fitted a
rank-8 correction from 256 paired, unlabeled activations and compared it with 20 residual-shuffle
fits, 20 source-shuffle fits, and rank-8 CORAL moment matching. AG News and MNLI were evaluated on
SmolLM-1.7B/SmolLM2-1.7B and independently trained Ai2/AMD OLMo 1B models, in both directions and
with seeds 42 and 137. This reuses previously studied tasks and is therefore diagnostic, not an
independent confirmation.

## Results

| Pair | Task | Paired recovery / retention | Residual-shuffle recovery | Pairing lift | Wins / $$p$$ | Decision |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| SmolLM | AG News | 0.803 / 0.827 | 0.605 | 0.196 | 4/4 / 0.0476 | pass |
| SmolLM | MNLI | 0.617 / 0.632 | 0.318 | 0.326 | 4/4 / 0.0476 | fail: retention |
| OLMo | AG News | 0.700 / 0.710 | 0.291 | 0.460 | 4/4 / 0.0476 | fail: retention |
| OLMo | MNLI | 0.354 / 0.372 | 0.150 | 0.221 | 4/4 / 0.0476 | fail: recovery and retention |

The paired correction beat all 20 residual shuffles in every seed-direction context (16/16), beat
the source-shuffle median in 16/16, and beat CORAL in 13/16. Across contexts, median paired
recovery was 0.665, retention 0.681, and pairing-specific lift 0.246. The minimum attainable
permutation probability with 20 shuffles was 1/21 in all contexts. Only one of four task-level
endpoints passed the complete locked rule because retention, and for OLMo/MNLI recovery, remained
insufficient.

Secondary confirmatory medians were AUROC 0.810, AUPRC 0.794, accuracy and balanced accuracy
0.700, precision 0.712, recall 0.750, F1 0.699, calibration error 0.231, and TPR 0.127/0.331 at
1%/5% FPR. The published artifacts retain thresholds, confusion counts, intervals, diagnostics,
and all row-level predictions.

## Conclusion and next step

The low-rank gain is reliably correspondence-dependent under these controls, but correspondence
does not guarantee adequate functional recovery. This supports task-conditioned, pairing-specific
repair rather than a universal canonical coordinate system. The next confirmatory study should
freeze this rank-8/256 protocol, increase null resolution, and evaluate prespecified new tasks.
