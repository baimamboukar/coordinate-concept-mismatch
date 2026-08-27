# August 27, 2026 | Attention-Head Permutation Probe Transport

[Hugging Face artifacts](https://huggingface.co/buckets/baimamboukar/coordinate-concept-mismatch/tree/studies/modern-attention-head-permutation-probe-transport/modern-attention-head-symmetry) | [W&B baseline](https://wandb.ai/JinesisLab/coordinate-concept-mismatch/runs/xcltbpyz) | [W&B symmetry](https://wandb.ai/JinesisLab/coordinate-concept-mismatch/runs/rnvpyg1p)

## Objective and method

This experiment tested whether an exact grouped-query-attention coordinate change is sufficient to break probe transfer. At block 24 of the pinned Mistral-7B-v0.3 checkpoint, we captured the concatenated attention-head representation immediately before `self_attn.o_proj`. Linear, degree-2 CP, and one-hidden-layer MLP probes were trained on WildGuardMix using data seeds 42 and 137, with 12,000 training, 2,000 validation, and 1,699 protected test examples.

For permutation seeds 42 and 137, we jointly permuted the eight key/value groups and their associated 32 query heads, then inversely permuted the corresponding input coordinates of the output projection. This preserves grouped-query associations, the attention-block output, and the model function while permuting the probed representation. We compared naive transfer, analytic probe transport, label-free exact feature matching from paired activations, and an inverse-transport control.

## Results

All three full function gates passed on the complete test set. Next-token agreement was 100%, maximum logit error was $2.17\times10^{-6}$, and measured attention-output equivariance error was zero.

| Condition | Mean AUROC | Mean AUPRC | Mean balanced accuracy | Mean TPR at 1% FPR |
| --- | ---: | ---: | ---: | ---: |
| Reference | 0.889 | 0.871 | 0.808 | 0.218 |
| Naive transfer | 0.660 | 0.599 | 0.540 | 0.052 |
| Analytic transport | 0.889 | 0.871 | 0.808 | 0.218 |
| Activation-estimated alignment | 0.889 | 0.871 | 0.808 | 0.218 |
| Inverse control | 0.696 | 0.665 | 0.623 | 0.109 |

The mean raw AUROC gap was 0.230, ranging from 0.136 to 0.348. All 12 probe-family, data-seed, and permutation-seed comparisons met the prespecified coordinate-failure rule, and every paired 95% bootstrap interval excluded zero. Analytic transport and activation-estimated alignment recovered 12/12 comparisons. All four label-free alignment maps exactly identified the planted head-coordinate permutation. Public artifacts retain the complete secondary metrics, thresholds, confusion counts, calibration values, diagnostics, and row-level predictions.

## Interpretation and next step

An exact attention-local coordinate mismatch is sufficient to produce substantial probe-transfer failure, including at low FPR. The smaller mean gap than in the MLP control indicates that sensitivity depends on the representation and symmetry. This result does not imply that independently trained attention representations differ only by head permutation. The next controlled intervention should test valid positive rescalings before normalization-compatible transformations and a second probe task.
