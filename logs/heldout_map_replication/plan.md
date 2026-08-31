# Held-Out Map Generalization | Independent-Pair Replication

## Objective

Replicate the OLMo finding that strong same-task affine recovery does not imply a task-general coordinate map. No tasks, thresholds, or probe hyperparameters will change after observing replication results.

## Model control

Use SmolLM-1.7B and SmolLM2-1.7B at step 5,125,000, the latest released intermediate checkpoint before context extension. Their pinned Hugging Face configurations match: 24 layers, width 2,048, 32 attention heads, tied embeddings, vocabulary 49,152, context 2,048, and RoPE base 10,000. The final SmolLM2 release is excluded because it changes RoPE. Both models use the same pinned SmolLM tokenizer; their token vocabularies and merges match.

The published [SmolLM recipe](https://github.com/huggingface/smollm/blob/a041759883ec7152d18fb985ea49be641a0bceef/text/pretraining/smollm1/config_smollm1_1B.yaml) and [SmolLM2 recipe](https://github.com/huggingface/smollm/blob/a041759883ec7152d18fb985ea49be641a0bceef/text/pretraining/smollm2/config_smollm2_1B.yaml) specify fresh initialization, with seeds 8 and 42. Data mixtures, training budgets, initialization scale, and schedules differ; this is independent-pretraining replication, not a pure seed intervention. Checkpoint selection is based on architecture compatibility, not probe performance.

## Fixed protocol

Retain the original pinned SST-2 and WildGuardMix fit tasks and AG News (World/Business) and MNLI (entailment/contradiction) held-out tasks. Each data seed, 42 and 137, uses 12,000 training and 2,000 validation examples. Protected tests contain 872, 1,699, 3,800, and 6,692 rows, respectively. Use raw prompts, the same 512-token limit, last-non-padding activations at 75% depth, and both model directions.

Linear probes are primary; degree-2 CP and MLP probes are secondary. Primary metrics are AUROC gap, recovered fraction, and retention of same-task improvement. Secondary metrics retain AUPRC, accuracy, balanced accuracy, precision, recall, F1, calibration, confusion counts, source-selected thresholds, achieved FPR, TPR at 1%/5% FPR, and row-level predictions. Use the unchanged 2,000-resample paired bootstrap with 95% intervals.

## Qualification and comparison

A task qualifies only if all four linear transfers fail under the existing rule (within-model AUROCs at least 0.75, gap at least 0.10, and positive lower confidence bound), and same-task affine recovery reaches 75% median with at least three substantial comparisons and no substantial shuffled control. Non-qualifying tasks remain reported and are not replaced.

Fit unlabeled maps on SST-2 alone, WildGuard alone, an equal pool (6,000 each), and a full pool (12,000 each). Compare affine and orthogonal maps plus shuffled-pair controls. Confirm included-task pooled compatibility before interpreting held-out degradation. The held-out criterion is at least 50% median recovery, 75% median retention, three substantial comparisons, and no substantial shuffled controls. A ten-point gain defines a material budget effect.

Broad replication requires both held-out tasks to qualify, included-task compatibility, and neither full-pool held-out condition to pass. Partial or contradictory outcomes narrow the claim. Heavy inputs and outputs stay on workers and Hugging Face; training is tracked in W&B. Verify publication before destroying workers.
