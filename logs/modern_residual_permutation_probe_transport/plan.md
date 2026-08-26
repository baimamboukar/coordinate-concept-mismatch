# August 26th, 2026 | Modern Residual-Permutation Probe Transport

[Hugging Face artifacts](https://huggingface.co/buckets/baimamboukar/coordinate-concept-mismatch/tree/studies/modern-residual-permutation-probe-transport/modern-models) | [Weights & Biases](https://wandb.ai/JinesisLab/coordinate-concept-mismatch)

## Objective

The Pythia control established that an exact residual-coordinate permutation can destroy frozen-probe performance without changing the language model's function. The natural modern-model experiments subsequently showed that label-free activation alignment usually recovers a substantial, but incomplete, fraction of cross-family transfer failure. This experiment connects those results by testing both analytic probe transport and activation-estimated alignment under a known symmetry of modern transformers. Mistral establishes the initial result; the prespecified Llama and Qwen extension tests whether it generalizes across architectures.

## Formal setup

Let $M$ be the pinned Mistral-7B-v0.3 checkpoint, $h_\ell(x) \in \mathbb{R}^{4096}$ its residual activation at 75% depth, and $P$ a seeded permutation matrix. We construct $M^P$ by consistently permuting the residual coordinates of the token embedding, RMSNorm parameters, attention and MLP input/output maps, final normalization, and language-model head. The transformation must satisfy

$$
M^P(x)=M(x), \qquad h^P_\ell(x)=P h_\ell(x).
$$

For each frozen probe $q$, we compare its reference score, naive score $q(h^P_\ell)$, analytically transported score $q^P(h^P_\ell)$, inverse-direction control, and score after a strict permutation alignment fitted only on paired, unlabeled training activations. The estimated alignment never observes task labels or the planted permutation.

## Protocol

- **Models:** `mistralai/Mistral-7B-v0.3`, `meta-llama/Llama-3.1-8B-Instruct`, and `Qwen/Qwen3-8B` at pinned repository revisions. Llama and Qwen run on isolated, model-scoped workers after the completed Mistral run.
- **Data:** the protected WildGuardMix split and data seeds 42 and 137 used in the frozen-transfer baseline.
- **Probes:** linear, degree-2 CP, and one-hidden-layer MLP probes at 75% depth.
- **Interventions:** identity plus residual permutations seeded by 42 and 137.
- **Function gates:** an eight-row fail-fast gate followed by all 1,699 protected test prompts in FP64; absolute logit tolerance $3\times10^{-5}$ plus relative tolerance $10^{-5}$, 100% next-token agreement, and activation-equivariance error at most $10^{-5}$. The absolute tolerance was fixed after a diagnostic full-set gate measured a maximum difference of $2.105\times10^{-5}$ with 100% next-token agreement; it accounts for Mistral's native float32 RMSNorm reduction while remaining negligible relative to probe-scale effects.
- **Estimated alignment:** strict feature permutation fitted on the first 2,000 paired training activations and diagnosed on all 2,000 validation activations.

## Outcomes and decision rule

Primary outcomes are the raw AUROC gap, analytic recovery fraction, and estimated-alignment recovery fraction. A comparison demonstrates coordinate-induced failure when reference AUROC is at least 0.75, the raw gap is at least 0.10, and its paired 95% bootstrap interval excludes zero. Analytic transport must recover at least 95% of the gap and reproduce reference scores within the prespecified tolerance. Estimated recovery is reported independently and succeeds at 95% recovery. Secondary outcomes retain AUROC, AUPRC, accuracy, balanced accuracy, precision, recall, F1, calibration, confusion counts, thresholds, TPR at 1% and 5% FPR, alignment diagnostics, and row-level predictions.

Failure of any function gate stops probe evaluation. The completed Mistral run establishes the modern-model control; Llama and Qwen determine whether the same causal result and label-free recovery hold across architectures.

For the cross-architecture extension, the same decision rule is applied separately to each model. Each model must pass all three full function gates, demonstrate coordinate-induced failure in every qualifying probe comparison, and recover at least 95% of the gap under analytic transport. Estimated alignment is evaluated independently. Results are published under model-specific artifact prefixes so concurrent workers cannot overwrite one another.
