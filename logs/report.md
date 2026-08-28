# August 28, 2026 | Experimental Progression

[Proposal](../docs/proposal.md) | [Frozen transfer](frozen_probe_transfer_baseline/report.md) | [Natural alignment](modern_activation_alignment_recovery/report.md) | [Residual symmetry](modern_residual_permutation_probe_transport/report.md) | [MLP permutation](modern_mlp_neuron_permutation_probe_transport/report.md) | [Attention symmetry](modern_attention_head_permutation_probe_transport/report.md) | [MLP rescaling](modern_mlp_positive_diagonal_probe_transport/report.md) | [Scale sweep](positive_diagonal_scale_sweep/report.md) | [Sentiment replication](sentiment_positive_diagonal_scale_sweep/report.md) | [OLMo 2 seeds](olmo2_same_architecture_seed_decomposition/report.md) | [Stage transition](olmo2_stage_transition_alignment/report.md) | [Independent training](olmo1_independent_training/report.md) | [Safety replication](olmo1_independent_training_wildguard/report.md) | [Cross-task transport](olmo1_cross_task_map_transport/report.md) | [Hugging Face](https://huggingface.co/buckets/baimamboukar/coordinate-concept-mismatch/tree/studies) | [W&B](https://wandb.ai/JinesisLab/coordinate-concept-mismatch)

The project asks whether frozen-probe transfer failure reflects incompatible coordinates or deeper representational differences.

Exact residual, MLP-neuron, and attention-component permutations first established sufficiency: behavior was preserved, naive transfer failed, and analytic transport restored the probes. Parameter symmetries can therefore cause transfer failure without changing model function.

Natural Pythia alignment recovered 60.3% of the median gap with restricted maps and about 97% with flexible maps. Across Llama, Qwen, Mistral, Granite, and Nemotron, all 20 cross-family comparisons failed; corresponding recovery was 61.8% and 97.3%, while shuffled maps recovered none. Flexible alignment demonstrates recoverability, not exact symmetry.

Positive-diagonal controls then isolated geometry and magnitude. Stronger scaling produced monotonic gaps concentrated in degree-2 probes, and the pattern replicated on SST-2. Probe fragility therefore depends jointly on symmetry magnitude, probe architecture, and task.

OLMo 2 supplied a modern same-architecture bridge. Stage-2 seed pairs had no primary late-layer failure, but Stage-1→Stage-2 transfer yielded six eligible failures. Permutation-plus-diagonal alignment recovered all six, with 77.0% median recovery; the effect vanished at earlier depths and in reverse directions that had not failed.

Architecture-identical Ai2 and AMD OLMo 1B checkpoints then produced four primary failures on both SST-2 and WildGuardMix. Median AUROC gaps were 0.406 and 0.461. Restricted recovery was 31.9% and 44.3%, whereas same-task affine recovery reached 99.1% and 94.9%; shuffled controls passed none. Dense alignment also restored low-FPR detection and calibration near each target-trained oracle.

Cross-task transport tested whether those dense maps represented a task-independent basis. SST-2→WildGuard recovered only 28.7% of the median gap with 30.1% same-task improvement retention; WildGuard→SST-2 recovered 37.4% with 37.9% retention. Only 3/8 comparisons were substantial, and the successful model direction reversed between tasks. Shuffled maps passed 0/8.

Together, the evidence shows that coordinate mismatch is sufficient and can explain substantial natural failure, but neither sparse symmetry-like recovery nor dense-map transport is universal. The next test should fit a task-balanced pooled map and evaluate a third held-out task, separating activation-support coverage from genuinely task-conditional alignment.
