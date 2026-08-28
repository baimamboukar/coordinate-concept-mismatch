# August 28, 2026 | OLMo 2 Same-Architecture Seed Decomposition

[Plan](plan.md) | [Hugging Face artifacts](https://huggingface.co/buckets/baimamboukar/coordinate-concept-mismatch/tree/studies/olmo2-same-architecture-seed-decomposition) | [W&B: WildGuard baseline](https://wandb.ai/JinesisLab/coordinate-concept-mismatch/runs/t5nkn5gk) | [WildGuard alignment](https://wandb.ai/JinesisLab/coordinate-concept-mismatch/runs/5p7m99zi) | [SST-2 baseline](https://wandb.ai/JinesisLab/coordinate-concept-mismatch/runs/nuwkrvgw) | [SST-2 alignment](https://wandb.ai/JinesisLab/coordinate-concept-mismatch/runs/lk1aaml8)

## Objective and protocol

We tested whether natural probe-transfer failure among architecture-identical models is attributable to coordinate mismatch. Three OLMo 2 1B Stage-2 checkpoints that diverged from a shared Stage-1 parent through training seed and data order were evaluated on WildGuardMix harmfulness and SST-2 sentiment. Frozen linear probes were primary; degree-2 CP and one-layer MLP probes were secondary. We used data seeds 42 and 137, four normalized depths, protected test splits, and paired bootstrap intervals. Alignment maps were fitted without labels or test examples.

## Findings

The prespecified 75%-depth linear result was negative: seed-to-seed transfer failed in 0/12 comparisons on both WildGuardMix and SST-2, with mean AUROC gaps of 0.002 and 0.016, respectively. WildGuardMix produced no failures at any tested depth or probe family.

SST-2 exposed a depth-localized effect. Linear mean gaps decreased from 0.082 at 25% depth to 0.043, 0.016, and 0.003 at 50%, 75%, and 100%; 5/12 early-layer comparisons met the prespecified failure criterion. For these five failures, permutation recovered 0% and permutation-plus-positive-diagonal alignment recovered 15.7% on average, with no substantial recoveries. Orthogonal Procrustes recovered 87.9%, while affine and quotient Ridge each recovered 78.6%; all five qualified as substantial under each flexible method. Shuffled affine alignment recovered none and reduced performance. Flexible recovery therefore demonstrates linear recoverability, not an exact parameter symmetry.

The Stage-1-to-Stage-2 lineage control produced 6/12 linear SST-2 failures at 25%, 50%, and 75% depth, but none at the final layer; WildGuardMix again produced none. This control was not included in the alignment stage and remains exploratory. All accepted linear fits converged, and the public artifacts retain primary and secondary metrics, thresholds, confusion counts, low-FPR statistics, and row-level predictions.

## Conclusion and next step

Training seed and data-order divergence did not create persistent late-layer probe mismatch in this shared-parent family, but early representations were task-dependent and fragile. The restricted maps explained little of the observed failure, whereas flexible maps recovered most of it. The next prespecified experiment should align the Stage-1-to-Stage-2 SST-2 failures, followed by models trained independently from initialization to test whether the conclusion survives stronger representational divergence.
