# Pythia Seed Probe Transfer

Status: completed on 2026-08-23.

## Prepared data

- Dataset: `allenai/wildguardmix` at
  `d29c47f41c8b51348b5c8e8c81c039b3132b66d1`.
- Seed 42: 12,000 balanced train rows and 2,000 balanced validation rows.
- Seed 137: 12,000 balanced train rows and 2,000 balanced validation rows.
- Protected test: 1,699 rows (945 unharmful and 754 harmful).
- Cleaning removed 38,924 duplicate train rows, 15 conflicting-label prompts,
  14 invalid train prompts, 3 train-test overlaps, and 26 unlabeled test rows.
- Prepared payload: 5,765,204 bytes; SHA-256
  `e1d2c873cd30214fbb99ee66276b729fc7aa45ab7804a940a5c38f66c9e62b11`.

Both pinned checkpoint tokenizers produced the same tokenization digest,
`e3be0dd47896e904f9cdab12ee5a7d75c49736ea139a027d8cfdbe603b6d661b`.
At the prespecified 512-token limit, truncation ranges from 0.18% on the protected
test set to 0.81% on the seed-42 training split, below the 5% gate.

## Local validation

Ruff, formatting, Pyright, and 39 tests pass. The tests include an end-to-end
synthetic coordinate reversal in which both oracle AUROCs are 1.0 and both frozen
cross-model probes fail with an AUROC gap of 1.0.

## Activation extraction

| Model | Split | Rows | Truncated | SHA-256 |
| --- | --- | ---: | ---: | --- |
| pythia_seed1234 | test | 1,699 | 0.18% | `55e5b37bbaf55a3b62d5359746ee583940ee80d0dd903a1d83d3cb09b21029ee` |
| pythia_seed1234 | seed_42_train | 12,000 | 0.81% | `d8b5fa1c6d68494b9947a4f760e2ad43f7e806fa120c2735a955eaf78494ee3c` |
| pythia_seed1234 | seed_42_validation | 2,000 | 0.60% | `38c2b39ba33fa8ed61eb1cb2d71eea0baad4889603eb8cb65ba28f0a650293fb` |
| pythia_seed1234 | seed_137_train | 12,000 | 0.77% | `ebaea669811014a31cfc5dc3a6a64a5cada3a4f4b8da339b5f19c049152ae6d6` |
| pythia_seed1234 | seed_137_validation | 2,000 | 0.70% | `451aa3a2aff52ded27881eb6901b07a74f3aa3e2c00c5af2ec23dce3986b74a8` |
| pythia_seed1 | test | 1,699 | 0.18% | `fa381031934e5b2aacb3e6fa39c6bc722ddd9bec30853449040b37f2a21ca75f` |
| pythia_seed1 | seed_42_train | 12,000 | 0.81% | `f747dea3c631ad49ec1fc86977e52817c3f3540faf82fe6de79da72626cb0296` |
| pythia_seed1 | seed_42_validation | 2,000 | 0.60% | `a2572b5d7ee03ff32e1d00db0731959899fc718828e886356a2132fae273088a` |
| pythia_seed1 | seed_137_train | 12,000 | 0.77% | `85de200b4fdc961cdb66a1dfa58c41ee21ef738804e2c6501592d9c7edf85718` |
| pythia_seed1 | seed_137_validation | 2,000 | 0.70% | `f1b2c7f1d7919b0628fb40fb084541237791765145f7e974703e88f61ea9840c` |

## Primary transfer results

| Data seed | Direction | Probe | Target oracle AUROC | Transfer AUROC | Gap | 95% CI | Prespecified failure |
| ---: | --- | --- | ---: | ---: | ---: | --- | --- |
| 42 | pythia_seed1234 → pythia_seed1 | linear | 0.864 | 0.442 | 0.422 | [0.385, 0.460] | yes |
| 42 | pythia_seed1234 → pythia_seed1 | cp_degree_2 | 0.865 | 0.540 | 0.325 | [0.293, 0.359] | yes |
| 42 | pythia_seed1234 → pythia_seed1 | mlp | 0.877 | 0.499 | 0.378 | [0.341, 0.413] | yes |
| 42 | pythia_seed1 → pythia_seed1234 | linear | 0.859 | 0.471 | 0.388 | [0.352, 0.425] | yes |
| 42 | pythia_seed1 → pythia_seed1234 | cp_degree_2 | 0.869 | 0.451 | 0.418 | [0.386, 0.449] | yes |
| 42 | pythia_seed1 → pythia_seed1234 | mlp | 0.875 | 0.504 | 0.372 | [0.338, 0.404] | yes |
| 137 | pythia_seed1234 → pythia_seed1 | linear | 0.855 | 0.406 | 0.450 | [0.415, 0.487] | yes |
| 137 | pythia_seed1234 → pythia_seed1 | cp_degree_2 | 0.834 | 0.572 | 0.262 | [0.230, 0.293] | yes |
| 137 | pythia_seed1234 → pythia_seed1 | mlp | 0.862 | 0.403 | 0.459 | [0.423, 0.494] | yes |
| 137 | pythia_seed1 → pythia_seed1234 | linear | 0.856 | 0.531 | 0.325 | [0.291, 0.357] | yes |
| 137 | pythia_seed1 → pythia_seed1234 | cp_degree_2 | 0.836 | 0.483 | 0.353 | [0.321, 0.385] | yes |
| 137 | pythia_seed1 → pythia_seed1234 | mlp | 0.865 | 0.503 | 0.363 | [0.330, 0.394] | yes |

All 12 primary comparisons meet the prespecified failure rule. Across them,
target-oracle AUROC is 0.834–0.877, frozen-transfer AUROC is 0.403–0.572, and
the paired AUROC gap is 0.262–0.459. Every paired 95% bootstrap interval
excludes zero. All 24 evaluated comparisons, including the linear probe at four
depths, meet the same rule.

## Secondary metrics

Values are medians over both data seeds and transfer directions at the primary
75% depth.

| Probe | Condition | AUROC | AUPRC | Accuracy | Balanced accuracy | Precision | Recall | F1 | ECE | TPR@1% FPR | TPR@5% FPR |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| linear | target oracle | 0.857 | 0.844 | 0.789 | 0.776 | 0.823 | 0.664 | 0.736 | 0.046 | 0.224 | 0.554 |
| linear | frozen transfer | 0.456 | 0.410 | 0.555 | 0.500 | 0.541 | 0.005 | 0.009 | 0.402 | 0.010 | 0.029 |
| cp_degree_2 | target oracle | 0.851 | 0.828 | 0.784 | 0.771 | 0.836 | 0.659 | 0.730 | 0.086 | 0.139 | 0.528 |
| cp_degree_2 | frozen transfer | 0.512 | 0.460 | 0.552 | 0.498 | 0.403 | 0.043 | 0.080 | 0.351 | 0.010 | 0.053 |
| mlp | target oracle | 0.870 | 0.853 | 0.801 | 0.789 | 0.841 | 0.683 | 0.753 | 0.091 | 0.198 | 0.560 |
| mlp | frozen transfer | 0.501 | 0.461 | 0.557 | 0.503 | 0.460 | 0.036 | 0.065 | 0.378 | 0.013 | 0.065 |

Per-condition confusion counts, thresholds, achieved operating points, and
row-level IDs, labels, scores, probabilities, and predictions are retained in
`results/metrics.jsonl` and `results/predictions.jsonl`.

## Staged artifacts

| File | SHA-256 |
| --- | --- |
| `activations/pythia_seed1/seed_137_train.safetensors` | `85de200b4fdc961cdb66a1dfa58c41ee21ef738804e2c6501592d9c7edf85718` |
| `activations/pythia_seed1/seed_137_validation.safetensors` | `f1b2c7f1d7919b0628fb40fb084541237791765145f7e974703e88f61ea9840c` |
| `activations/pythia_seed1/seed_42_train.safetensors` | `f747dea3c631ad49ec1fc86977e52817c3f3540faf82fe6de79da72626cb0296` |
| `activations/pythia_seed1/seed_42_validation.safetensors` | `a2572b5d7ee03ff32e1d00db0731959899fc718828e886356a2132fae273088a` |
| `activations/pythia_seed1/test.safetensors` | `fa381031934e5b2aacb3e6fa39c6bc722ddd9bec30853449040b37f2a21ca75f` |
| `activations/pythia_seed1234/seed_137_train.safetensors` | `ebaea669811014a31cfc5dc3a6a64a5cada3a4f4b8da339b5f19c049152ae6d6` |
| `activations/pythia_seed1234/seed_137_validation.safetensors` | `451aa3a2aff52ded27881eb6901b07a74f3aa3e2c00c5af2ec23dce3986b74a8` |
| `activations/pythia_seed1234/seed_42_train.safetensors` | `d8b5fa1c6d68494b9947a4f760e2ad43f7e806fa120c2735a955eaf78494ee3c` |
| `activations/pythia_seed1234/seed_42_validation.safetensors` | `38c2b39ba33fa8ed61eb1cb2d71eea0baad4889603eb8cb65ba28f0a650293fb` |
| `activations/pythia_seed1234/test.safetensors` | `55e5b37bbaf55a3b62d5359746ee583940ee80d0dd903a1d83d3cb09b21029ee` |
| `probes/seed_137/pythia_seed1.safetensors` | `c5d47a6c716200d45b6ddbe3007c33740f3f0cbb8d66a69688e7cce9c81ceaac` |
| `probes/seed_137/pythia_seed1234.safetensors` | `7dc3879d53d3fc241507562366fdcaf413c857b8381d5282ce996e3d8572c230` |
| `probes/seed_42/pythia_seed1.safetensors` | `48bceadfd003fc27130df429c5290666018f40bb5734a31e9ad84b715e136cae` |
| `probes/seed_42/pythia_seed1234.safetensors` | `d3e3121527c88c87d1d170dbd92d781184126ddda7e92e5693aae95cdb471a2a` |
| `results/metrics.jsonl` | `49f2b8f73bd4fdf719bd7e30f1b09c401c30c03d91f527067638d29ec9e352f8` |
| `results/predictions.jsonl` | `27149a2a6dd9130b694a7b2b1b66af57b339ae40cbebb481d9225e9213204530` |
| `results/transfer_gaps.jsonl` | `8ed0af93096ff7ddf670c4596df0a71d024d2152b60d16a9d8ecb034c33cc40c` |

## Completion

Extracted 59,398 model-row pairs and completed probe training on Vast.ai H100
instance `48499249`, labeled `824061`, at $2.37037/hour including disk. The job
exited with code 0. The retrieved 506,869,760-byte archive has SHA-256
`524ef09d620d16c05fe2fcb7a386d6d1362e7518fb69b9f385ff8f2a1267a55f`.
The instance was destroyed and verified absent from the provider inventory.

All 17 derived artifacts were uploaded to
`hf://buckets/baimamboukar/coordinate-concept-mismatch/experiments/pythia_seed_probe_transfer`.
A complete download round trip reproduced every recorded checksum and byte
count (506,530,672 bytes). The gated prompt text and offline W&B files were not
uploaded. The offline training run was synced and verified finished at
[Weights & Biases](https://wandb.ai/JinesisLab/coordinate-concept-mismatch/runs/hd5rdpqo).

## Interpretation

This experiment establishes a large raw frozen-probe transfer failure between
two independently trained Pythia-410M checkpoints with identical architecture
and training budget. Increased probe capacity does not remove the failure in
this setting. The result does not establish that probes generally fail across
modern model families, and it does not determine whether this gap is caused by
coordinate mismatch or genuinely different learned representations.

## Next step

Apply an exactly function-preserving symmetry transformation to one checkpoint,
verify output equivalence, and measure probe transfer before and after exact
probe transport. Then compare those controlled effects with symmetry-aware
alignment between the two independent checkpoints.
