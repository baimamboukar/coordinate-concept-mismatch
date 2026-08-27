# August 27, 2026 | Sentiment Positive-Diagonal Scale Sweep

[Hugging Face artifacts](https://huggingface.co/buckets/baimamboukar/coordinate-concept-mismatch/tree/studies/positive-diagonal-task-generalization/sentiment-positive-diagonal-scale-sweep) | [Weights & Biases](https://wandb.ai/JinesisLab/coordinate-concept-mismatch)

## Objective

The WildGuardMix scale sweep found a monotonic coordinate-magnitude effect, with robust failures confined to degree-2 CP probes at the strong and extreme ranges. This experiment tests whether that probe-family sensitivity generalizes to a different binary concept: sentence sentiment. Because the family-specific hypothesis was derived from WildGuardMix, this is a targeted out-of-task replication rather than an independent discovery.

## Formal setup

For the Mistral block-24 post-SwiGLU activation $h(x)$ and output projection $W_{\mathrm{down}}$, each positive diagonal map $D_r$ defines

$$
h^{D_r}(x)=D_rh(x),
\qquad
W_{\mathrm{down}}^{D_r}=W_{\mathrm{down}}D_r^{-1}.
$$

The model function is unchanged, while the frozen probe receives a different coordinate representation. We retain the paired mild $[1/2,2]$, moderate $[1/8,8]$, strong $[1/32,32]$, and extreme $[1/128,128]$ ranges from the preceding experiment.

## Protocol

- **Task:** positive versus negative sentiment from the pinned `stanfordnlp/sst2` dataset. The 872 labeled validation examples form the protected test set; deterministic, label-balanced train and validation samples contain 12,000 and 2,000 examples for seeds 42 and 137.
- **Fixed factors:** Mistral-7B-v0.3, block 24, last non-padding token, transformation seeds 42 and 137, and the same linear, degree-2 CP, and one-hidden-layer MLP probe selection procedures.
- **Controls:** reference, identity, naive frozen transfer, analytic transport, label-free diagonal estimation, and inverse transport. Eight-prompt smoke gates precede FP64 gates over the full protected test set.
- **Artifacts:** complete metrics, thresholds, confusion counts, calibration, TPR at 1% and 5% FPR, row-level predictions, transported probes, and alignment diagnostics are published directly from the worker.

## Decision rules

Task viability requires every reference probe AUROC to be at least 0.75; otherwise family-generalization is inconclusive and results remain descriptive. The existing dose-response rule is unchanged: nondecreasing mean gaps and Spearman $\rho\geq0.8$ for at least 10 of 12 trajectories.

Degree-2 sensitivity generalizes only if, at both strong and extreme ranges, at least 3 of 4 CP comparisons fail, at most 2 of 8 linear/MLP comparisons fail, and the CP mean AUROC gap exceeds the larger comparison-family mean by at least 0.05. Coordinate failure still requires reference AUROC at least 0.75, raw gap at least 0.10, and a positive paired 95% bootstrap lower bound. Analytic and estimated recovery must reach 95% wherever the raw gap is positive.
