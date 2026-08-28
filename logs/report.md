# August 28, 2026 | Experimental Progression

[Proposal](../docs/proposal.md) | [Frozen transfer](frozen_probe_transfer_baseline/report.md) | [Natural alignment](modern_activation_alignment_recovery/report.md) | [Symmetry controls](modern_residual_permutation_probe_transport/report.md) | [Scale sweep](positive_diagonal_scale_sweep/report.md) | [Sentiment replication](sentiment_positive_diagonal_scale_sweep/report.md) | [OLMo 2 seeds](olmo2_same_architecture_seed_decomposition/report.md) | [Stage transition](olmo2_stage_transition_alignment/report.md) | [Independent training](olmo1_independent_training/report.md) | [Safety replication](olmo1_independent_training_wildguard/report.md) | [Cross-task transport](olmo1_cross_task_map_transport/report.md) | [Held-out BoolQ](olmo1_pooled_map_heldout_boolq/report.md) | [Pooled compatibility](olmo1_pooled_map_compatibility/report.md) | [Hugging Face](https://huggingface.co/buckets/baimamboukar/coordinate-concept-mismatch/tree/studies) | [W&B](https://wandb.ai/JinesisLab/coordinate-concept-mismatch)

The project asks whether frozen-probe transfer failure reflects incompatible coordinates or deeper representational differences.

Exact residual, MLP-neuron, and attention-component permutations established sufficiency: behavior was preserved, naive transfer failed, and analytic transport restored the probes. Parameter symmetries can therefore cause failure without changing model function.

Natural Pythia alignment recovered 60.3% of the median gap with restricted maps and about 97% with flexible maps. Across five modern families, all 20 comparisons failed; corresponding recovery was 61.8% and 97.3%, while shuffled maps recovered none. Flexible alignment demonstrates recoverability, not exact symmetry.

Positive-diagonal controls isolated magnitude: stronger scaling produced monotonic gaps concentrated in degree-2 probes, and the pattern replicated on SST-2. Fragility therefore depends on symmetry magnitude, probe architecture, and task.

OLMo 2 supplied a same-architecture bridge. Stage-2 seed pairs had no primary late-layer failure, but Stage-1→Stage-2 yielded six eligible failures. Permutation-plus-diagonal alignment recovered all six, with 77.0% median recovery; the effect vanished at earlier depths and in non-failing reverse directions.

Architecture-identical Ai2 and AMD OLMo 1B checkpoints produced four failures on both SST-2 and WildGuardMix. Median AUROC gaps were 0.406 and 0.461. Restricted recovery was 31.9% and 44.3%, whereas same-task affine recovery reached 99.1% and 94.9%; shuffled controls passed none.

Single-task map transport was weak: SST-2→WildGuard recovered 28.7% of the median gap and WildGuard→SST-2 recovered 37.4%. Only 3/8 comparisons were substantial; shuffled maps passed 0/8.

BoolQ then failed the held-out qualification gate: target-trained AUROC was only 0.639–0.646, below the prespecified 0.75 threshold, so no alignment maps were fit.

A subsequent compatibility test fitted one balanced map on SST-2 and WildGuardMix. At equal total budget, median affine recovery was 83.0% on SST-2 and 94.1% on WildGuardMix, with 3/4 and 4/4 substantial comparisons; shuffled controls passed 0/8. Doubling data raised recovery only to 85.7% and 94.8%. Thus, poor single-task transport mainly reflected concept coverage, not evidence of incompatible coordinates. Because pooled fitting included both evaluation tasks, the next decisive test is transfer without refitting to a preregistered, decodable held-out panel.
