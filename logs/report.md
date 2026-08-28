# August 28, 2026 | Experimental Progression

[Proposal](../docs/proposal.md) | [Frozen transfer](frozen_probe_transfer_baseline/report.md) | [Natural alignment](modern_activation_alignment_recovery/report.md) | [Symmetry controls](modern_residual_permutation_probe_transport/report.md) | [Scale sweep](positive_diagonal_scale_sweep/report.md) | [Sentiment replication](sentiment_positive_diagonal_scale_sweep/report.md) | [OLMo 2 seeds](olmo2_same_architecture_seed_decomposition/report.md) | [Stage transition](olmo2_stage_transition_alignment/report.md) | [Independent training](olmo1_independent_training/report.md) | [Safety replication](olmo1_independent_training_wildguard/report.md) | [Cross-task transport](olmo1_cross_task_map_transport/report.md) | [Held-out BoolQ](olmo1_pooled_map_heldout_boolq/report.md) | [Pooled compatibility](olmo1_pooled_map_compatibility/report.md) | [Held-out panel](olmo1_heldout_task_panel/report.md) | [Hugging Face](https://huggingface.co/buckets/baimamboukar/coordinate-concept-mismatch/tree/studies) | [W&B](https://wandb.ai/JinesisLab/coordinate-concept-mismatch)

The project asks whether frozen-probe transfer failure reflects incompatible coordinates or representational differences.

Exact residual, MLP-neuron, and attention-component permutations established sufficiency: behavior was preserved, naive transfer failed, and analytic transport restored probes. Symmetries can therefore cause failure without changing model function.

Natural Pythia alignment recovered 60.3% of the median gap with restricted maps and about 97% with flexible maps. Across five modern families, all 20 comparisons failed; recovery was 61.8% and 97.3%, while shuffled maps recovered none. Flexible alignment demonstrates recoverability, not exact symmetry.

Positive-diagonal controls produced monotonic gaps concentrated in degree-2 probes, replicated on SST-2. Fragility depends on symmetry magnitude, probe architecture, and task.

OLMo 2 supplied a same-architecture bridge. Stage-2 seed pairs had no primary late-layer failure, but Stage-1→Stage-2 yielded six; permutation-plus-diagonal alignment recovered all six with 77.0% median recovery. The effect vanished at earlier depths and in reverse directions.

Architecture-identical Ai2 and AMD OLMo 1B checkpoints produced four failures on both SST-2 and WildGuardMix. Median gaps were 0.406 and 0.461. Restricted recovery was 31.9% and 44.3%, versus same-task affine recovery of 99.1% and 94.9%; shuffled controls passed none.

Single-task map transport was weak: SST-2→WildGuard recovered 28.7% and WildGuard→SST-2 37.4%; 3/8 comparisons were substantial and shuffled maps passed 0/8.

BoolQ failed qualification because target-trained AUROC was 0.639–0.646, so no maps were fit.

A balanced SST-2/WildGuard map recovered 83.0% and 94.1% on those included tasks; doubling data added little. The preregistered held-out panel then qualified AG News and MNLI: frozen median gaps were 0.505 and 0.410, while same-task affine recovery reached 98.6% and 96.1%. Yet equal-pool maps recovered only 60.4% and 32.3%, and full-pool maps 57.1% and 35.9%; no pooled condition passed, and 0/32 shuffled controls did. Included-task compatibility therefore does not establish a global coordinate system: for this model pair, flexible alignment is strongly concept-distribution-bound.
