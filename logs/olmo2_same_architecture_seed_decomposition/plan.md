# Modern Same-Architecture Seed Decomposition

## Objective

Measure how much frozen-probe transfer failure between naturally diverged, architecture-identical language models is recovered by increasingly flexible activation alignment. This supplies the missing bridge between exact planted symmetries and the completed cross-family experiments.

## Model lineage

The experiment uses the 1B-parameter OLMo 2 architecture: 16 layers and a 2,048-dimensional residual stream. Three public Stage-2 runs begin from the same Stage-1 checkpoint and continue for approximately 51B tokens under seeds 42069, 666, and 42 with different data order. Their common Stage-1 checkpoint is retained only as a lineage control.

These models are independent Stage-2 trajectories, not independent pretraining runs from random initialization. Conclusions therefore concern training-seed and data-order divergence conditional on a shared parent.

## Formal comparison

For source seed $s$, target seed $t$, layer $\ell$, and frozen source probe $p_s$, evaluate the identity condition and an unlabeled target-to-source map $A$:

$$
S_{s\rightarrow t}(A)
=
\operatorname{AUROC}\!\left(p_s(Ah_t^\ell(x)),y\right).
$$

Relative to raw transfer $S_{s\rightarrow t}(I)$ and the target-trained oracle $S_{t\rightarrow t}$, recovery is

$$
R(A)
=
\frac{S_{s\rightarrow t}(A)-S_{s\rightarrow t}(I)}
{S_{t\rightarrow t}-S_{s\rightarrow t}(I)}.
$$

Maps are fitted on paired, unlabeled training prompts. Validation activations diagnose alignment; labels and protected test examples are excluded from fitting.

## Protocol

- **Tasks:** WildGuardMix prompt harmfulness is primary; SST-2 sentiment is the task replication.
- **Representations:** final non-padding token at 25%, 50%, 75%, and 100% depth; 75% is primary.
- **Probes:** linear logistic regression is primary; degree-2 CP and one-hidden-layer MLP probes are secondary.
- **Repetitions:** data seeds 42 and 137, with one protected test split per task.
- **Alignment ladder:** identity, permutation, permutation plus positive diagonal, orthogonal Procrustes, affine Ridge, quotient Ridge, and target oracle. Shuffled-pair affine alignment is the negative control.

A transfer failure requires source and target-oracle AUROC of at least 0.75, an AUROC gap of at least 0.10, and a paired 95% bootstrap interval excluding zero. Substantial restricted recovery additionally requires improvement of at least 0.05 and recovery of at least 50%. Flexible recovery is interpreted as linear recoverability, not parameter symmetry.

Primary outcomes are AUROC transfer gap, restricted recovery fraction, and residual oracle gap. Secondary outcomes include AUPRC, accuracy, balanced accuracy, precision, recall, F1, calibration error, confusion counts, TPR at 1% and 5% FPR, achieved FPR under source thresholds, alignment RMSE/cosine, all retained thresholds, and row-level scores and predictions.

All bulky activations and results publish directly from the GPU worker to the project Hugging Face bucket. Probe training is tracked in W&B. No activation tensor is routed through the coordinating machine.
