# Pythia Residual-Permutation Probe Transport

## Objective

Experiment 1 established frozen-probe transfer failure between independently trained Pythia-410M checkpoints, but could not separate coordinate mismatch from representational difference. This experiment asks whether a known residual-coordinate change alone can cause that failure and whether exact probe transport repairs it.

## Formal setup

Let $M$ be a frozen model, $h_\ell(x) \in \mathbb{R}^{1024}$ its residual activation, and $q$ a probe including its frozen preprocessing. For a permutation matrix $P$, construct $M^P$ by transforming the embedding, LayerNorm parameters, attention and MLP maps, and output head so that

$$
M^P(x)=M(x),
\qquad
h^P_\ell(x)=P h_\ell(x).
$$

The transported probe is defined by

$$
q^P(z)=q(P^{-1}z),
\qquad
q^P(h^P_\ell(x))=q(h_\ell(x)).
$$

This intervention changes the coordinates observed by the probes while preserving the model's function. MLP-neuron permutations are not used here because they leave the block-level residual states from Experiment 1 unchanged.

## Design

- **Models:** `EleutherAI/pythia-410m` is primary; `EleutherAI/pythia-410m-seed1` is a prespecified replication. Both retain the revisions used in Experiment 1.
- **Interventions:** an identity control and two independently sampled global residual permutations, with seeds 42 and 137.
- **Materials:** reuse both data seeds, the shared 1,699-row test set, and the frozen probes from the Pythia pilot reported in [Experiment 1](../frozen_probe_transfer_baseline/report.md).
- **Probes:** linear, degree-2 CP, and one-hidden-layer MLP probes at block 18 are primary; linear probes at blocks 6, 12, and 24 are secondary.
- **Conditions:** original reference, identity control, raw frozen transfer, exact probe transport, and inverse-direction transport.

Probe parameters, preprocessing, thresholds, examples, and labels remain fixed. No target labels or retraining enter the primary analysis.

## Outcomes and decision rules

**H1 — Function preservation.** In deterministic FP64 over all 1,699 held-out prompts at the final non-padding position, the transformed logits must satisfy `atol=1e-8, rtol=1e-8`, with 100% next-token agreement, and relative activation-equivariance error at most $10^{-10}$ at every probed layer. Failure of either gate invalidates the intervention.

**H2 — Coordinate-induced failure.** With reference AUROC $A_{\mathrm{ref}}$ and raw-transfer AUROC $A_{\mathrm{raw}}$, define $G_{\mathrm{raw}}=A_{\mathrm{ref}}-A_{\mathrm{raw}}$. Failure requires $A_{\mathrm{ref}} \geq 0.75$, $G_{\mathrm{raw}} \geq 0.10$, and a paired 95% bootstrap interval whose lower bound exceeds zero.

**H3 — Exact recovery.** For transported AUROC $A_{\mathrm{transport}}$, define

$$
R=\frac{A_{\mathrm{transport}}-A_{\mathrm{raw}}}
{A_{\mathrm{ref}}-A_{\mathrm{raw}}}.
$$

When $G_{\mathrm{raw}}>0$, recovery requires transported scores to match reference scores within `atol=1e-4, rtol=1e-5`, AUROC within 0.01 of reference, and $R \geq 0.95$. H2 and H3 are assessed separately by probe family and must hold for both permutation seeds and both data seeds on the primary checkpoint. The replication checkpoint is reported independently.

Primary outcomes are raw AUROC gap and recovery fraction. Secondary outcomes are AUPRC, accuracy, balanced accuracy, precision, recall, F1, calibration error, `tn/fp/fn/tp`, TPR at 1% and 5% FPR, threshold transfer, score agreement, and layerwise results. Row-level IDs, labels, scores, probabilities, predictions, and thresholds are retained.

## Workflow

1. Implement and unit-test the residual-basis transformation and transport of every probe and preprocessor.
2. Pass the identity, function-preservation, and activation-equivariance gates before evaluating transfer.
3. Run both permutations on the primary checkpoint without changing the protocol.
4. Repeat the accepted protocol on the replication checkpoint.
5. Bootstrap paired gaps, track the run in W&B, upload derived artifacts to Hugging Face, and write a concise dated report.

## Interpretation boundary

Raw failure followed by exact recovery would show that coordinate mismatch is sufficient to cause probe-transfer failure. It would not estimate how much of the natural gap between independent checkpoints is coordinate-based; the following alignment experiment will address that question. Transport failure after the equivalence gates pass is treated as an implementation failure, not evidence of concept mismatch.
