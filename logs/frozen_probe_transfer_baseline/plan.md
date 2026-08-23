# Frozen Probe Transfer Baseline

> Establish whether a probe that is valid within one modern model loses predictive performance when transferred unchanged to another model.

## Research question

For a prompt $x$ with harmfulness label $y$, let $h_m^\ell(x) \in \mathbb{R}^{4096}$ be the final-prompt-token activation from model $m$ at normalized depth $\ell$. For probe family $a$, a source probe $p_s^a$ is trained on source-model activations and then frozen. We compare its target-model performance with a matched probe trained directly on the target:

$$
G_{s\rightarrow t}^{a}
=
\operatorname{AUROC}(p_t^a(h_t))
-
\operatorname{AUROC}(p_s^a(h_t)).
$$

The experiment tests whether $G_{s\rightarrow t}^{a}$ is reliably positive. It does **not** determine whether a gap is caused by coordinate mismatch or representation mismatch; that is the purpose of later controlled-symmetry and alignment experiments.

## Models

All selected checkpoints have 4,096-dimensional residual streams, so applying a frozen probe without an adapter is numerically defined. Every revision is pinned to the current Hugging Face commit.

| Role                  | Model                                                                                        | Layers | Revision                                   |
| --------------------- | -------------------------------------------------------------------------------------------- | -----: | ------------------------------------------ |
| Independent family    | [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)             |     32 | `0e9e39f249a16976918f6564b8830bc894c89659` |
| Independent family    | [Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)        |     32 | `c170c708c41dac9275d15a8fff4eca08d52bab71` |
| Independent family    | [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)                                             |     36 | `b968826d9c46dd6066d109eabc6255188de91218` |
| Llama-lineage control | [Llama-3.1-Nemotron-Nano-8B-v1](https://huggingface.co/nvidia/Llama-3.1-Nemotron-Nano-8B-v1) |     32 | `54641c1611fcff44fa4865626462445e0a153fc7` |

Nemotron is derived from Llama 3.1 Instruct and will therefore be reported as a lineage comparison, not as an independent model family.

## Data and prediction task

We use [WildGuardMix](https://huggingface.co/datasets/allenai/wildguardmix), pinned at `d29c47f41c8b51348b5c8e8c81c039b3132b66d1`. The binary target is `prompt_harm_label`: `harmful` versus `unharmful`.

- Training source: `wildguardtrain/train` (86,759 released rows).
- Final evaluation: `wildguardtest/test` (1,725 released rows).
- Remove rows with missing prompt-harm labels.
- Deduplicate normalized prompts within and across splits, protecting the test split.
- Freeze the dataset revision, cleaning rules, sampling code, and pilot seeds before activation extraction.
- Draw 12,000 balanced training and 2,000 validation prompts, stratified by label and adversarial status where feasible, under two fixed pilot seeds.
- Keep the cleaned test set untouched and report results separately for vanilla and adversarial prompts.

The train labels were primarily produced using GPT-4 and audited on a human-labeled sample; the test labels use three independent annotators. This measurement difference must be recorded as a limitation.

## Probe families and training

We use a controlled capacity ladder rather than treating one probe architecture as representative of all probes.

1. **Linear probe — primary.** L2-regularized logistic regression is the lowest-capacity readout and the cleanest test of whether the same decision direction survives across models. Its regularization strength is selected from a fixed grid using source validation AUROC.
2. **Low-rank degree-2 CP probe — confirmatory nonlinear test.** A full quadratic is impractical at width 4,096: it has 8,394,753 coefficients. We instead use the affine-completed rank-$r$ parameterization

$$
p_{r}(h)
=
\sum_{j=1}^{r}
\alpha_j
(\langle u_j,h\rangle+u_{j,0})
(\langle v_j,h\rangle+v_{j,0})
+\langle w,h\rangle+w_0.
$$

Ranks $r\in\{1,2,4,8,16\}$ and weight decay are selected using only source validation data. This family tests pairwise feature interactions while remaining tractable and structurally stable under affine coordinate changes. 3. **One-hidden-layer MLP — exploratory capacity control.** A width-32 GELU MLP has approximately 131,000 parameters, close to the rank-16 CP probe's 135,217. It tests whether generic nonlinear capacity changes the conclusion, but it is not an affine-stable probe family and is not part of the confirmatory decision rule.

All probes use frozen activations. Preprocessing is fit on source-training activations and frozen for transfer. Linear probes use logistic loss and LBFGS. CP and MLP probes use binary cross-entropy, AdamW, early stopping on source validation AUROC, and three initialization restarts per data seed. No target labels may influence source-probe selection, preprocessing, early stopping, or thresholds.

Degree $3+$ probes are deferred. They are justified only if degree 2 adds meaningful target-oracle performance but leaves transfer failure unresolved; otherwise they add capacity without answering a new question.

## Experimental workflow

1. Audit labels, duplicates, class balance, prompt lengths, and train-test overlap.
2. Tokenize the same raw prompt text with each model's tokenizer. The primary condition uses no chat template; native chat templates form a secondary sensitivity analysis.
3. Set `max_length=512`. If any tokenizer truncates more than 5% of prompts, increase the cap before extraction and report the final truncation rate per model.
4. Extract the model-reported final non-padding prompt-token hidden state at 25%, 50%, 75%, and 100% of model depth. The prespecified primary layer is 75%; other depths are secondary. At 100% depth, this includes the architecture's terminal normalization when Transformers applies it before returning the final hidden state.
5. Fit every prespecified probe family on source-training activations. Select hyperparameters and stopping points only on the source validation split. Freeze the source-derived preprocessing, probe weights, and decision threshold.
6. Evaluate each frozen source probe on every compatible target model. Separately train the same architecture directly on each target to measure target decodability.
7. Compare train, source-validation, source-test, transfer-test, and target-oracle performance so that probe underfitting and overfitting cannot be mistaken for transfer failure.
8. Repeat across the two fixed pilot seeds. Use paired bootstrap confidence intervals over identical test prompts.

## Metrics and decision rule

The primary metric is target AUROC at 75% depth. Secondary metrics are AUPRC, accuracy, balanced accuracy, precision, recall, F1, expected calibration error, full `tn/fp/fn/tp`, and TPR at 1% and 5% FPR. Threshold-dependent metrics use a threshold selected only on source validation data and frozen for transfer. We also record the achieved target FPR and TPR at source-selected 1% and 5% FPR thresholds. Row-level IDs, labels, scores, probabilities, and predictions, together with every selected threshold, must be retained for every probe family.

A directed transfer for probe family $a$ is classified as failed only when:

1. both the source and target-trained probes achieve AUROC $\geq 0.75$;
2. the transfer gap is at least $0.10$; and
3. the paired 95% bootstrap confidence interval for the gap excludes zero.

The primary claim is **broad linear-probe transfer failure**. We will make it only if at least four of the six directed Llama-Mistral-Qwen transfers satisfy this rule and their median gap is at least $0.10$. We will make the stronger claim **broad shallow-probe transfer failure** only if at least four directed transfers fail for both the linear and selected degree-2 CP probes. Otherwise, the result will be reported as architecture-dependent or mixed transfer. MLP and Nemotron results are analyzed separately.
