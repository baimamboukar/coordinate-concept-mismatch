# August 31, 2026 | Experimental Progression

[Proposal](../docs/proposal.md) | [Frozen transfer](frozen_probe_transfer_baseline/report.md) | [Natural alignment](modern_activation_alignment_recovery/report.md) | [Symmetry controls](modern_residual_permutation_probe_transport/report.md) | [Scale sweep](positive_diagonal_scale_sweep/report.md) | [Sentiment replication](sentiment_positive_diagonal_scale_sweep/report.md) | [OLMo 2 seeds](olmo2_same_architecture_seed_decomposition/report.md) | [Stage transition](olmo2_stage_transition_alignment/report.md) | [Independent training](olmo1_independent_training/report.md) | [Safety replication](olmo1_independent_training_wildguard/report.md) | [Cross-task transport](olmo1_cross_task_map_transport/report.md) | [Held-out BoolQ](olmo1_pooled_map_heldout_boolq/report.md) | [Pooled compatibility](olmo1_pooled_map_compatibility/report.md) | [Held-out panel](olmo1_heldout_task_panel/report.md) | [SmolLM replication](heldout_map_replication/report.md) | [Shared-map compatibility](shared_map_compatibility/report.md) | [Hugging Face](https://huggingface.co/buckets/baimamboukar/coordinate-concept-mismatch/tree/studies) | [W&B — private](https://wandb.ai/JinesisLab/coordinate-concept-mismatch)

The project asks whether frozen-probe transfer failure reflects incompatible coordinates or representational differences.

Exact residual, MLP-neuron, and attention-component permutations preserved behavior, broke naive transfer, and permitted analytic probe recovery. This established that coordinate changes can cause failure without changing model function.

Natural baselines, piloted on Pythia and expanded to five modern families, then tested practical relevance. All 20 modern comparisons failed; restricted and flexible maps recovered 61.8% and 97.3%, while shuffled maps recovered none. Recoverability does not establish exact symmetry.

Positive-diagonal controls produced monotonic gaps concentrated in degree-2 probes, replicated on SST-2: probe architecture and transformation magnitude matter.

OLMo 2 separated lineage effects. Stage-2 seeds had no primary late-layer failure, whereas Stage-1→Stage-2 yielded six, with 77.0% median restricted recovery. The effect vanished at earlier depths and in reverse directions.

Independent Ai2/AMD OLMo 1B training produced four failures each on SST-2 and WildGuard. Restricted recovery was 31.9%/44.3%, versus same-task affine recovery of 99.1%/94.9%.

To test whether this represented a shared coordinate map, we transferred maps across tasks. Recovery fell to 28.7%/37.4%. Pooling restored included-task compatibility to 83.0%/94.1%, motivating held-out evaluation.

BoolQ failed qualification and was not replaced post hoc. The preregistered AG News/MNLI panel subsequently qualified: same-task recovery was 98.6%/96.1%, but full-pool recovery fell to 57.1%/35.9%. No pooled condition passed; doubling data had no material effect.

The independent SmolLM replication qualified both held-out tasks, with 96.6%/97.7% same-task recovery and 72.5%/53.4% full-pool recovery. However, AG News narrowly missed the retention cutoff; additional data materially helped MNLI; and pooling failed included SST-2 compatibility, recovering only 66.9%. Thus this is partial replication, not confirmation of the isolated OLMo held-out effect.

The fitting diagnostic then exposed a trade-off: scale-balanced, validation-selected ridge recovered 96.6% on SST-2 but only 64.6% on WildGuard. No condition passed both compatibility gates; held-out evaluation was skipped. Weighting redistributed performance, while regularization selection changed little within the tested grid.

Next, separate alignment calibration from probe-training data and reserve untouched evaluation data. Existing overlap and exploratory test reuse limit interpretation; residual transfer loss cannot establish intrinsic concept mismatch.
