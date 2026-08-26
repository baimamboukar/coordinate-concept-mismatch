# August 26th, 2026 | Modern Residual-Permutation Probe Transport

[Hugging Face artifacts](https://huggingface.co/buckets/baimamboukar/coordinate-concept-mismatch/tree/studies/modern-residual-permutation-probe-transport/modern-models) | [Weights & Biases](https://wandb.ai/JinesisLab/coordinate-concept-mismatch/runs/mh826137)

## Result

We applied two seeded global residual-coordinate permutations to the pinned Mistral-7B-v0.3 checkpoint and evaluated frozen probes at 75% depth. Both transformations passed the full function-preservation gate on all 1,699 protected test prompts: next-token agreement was 100%, the maximum logit difference was $2.105\times10^{-5}$, and the maximum activation-equivariance error was $1.379\times10^{-7}$.

Across data seeds 42 and 137, permutation seeds 42 and 137, and linear, degree-2 CP, and one-hidden-layer MLP probes, all 12 comparisons met the prespecified coordinate-failure rule. Mean AUROC fell from 0.912 for reference probes to 0.469 under naive transfer, a mean gap of 0.442; individual gaps ranged from 0.359 to 0.530, and every paired 95% bootstrap interval excluded zero.

Analytic probe transport restored the reference AUROC in all 12 comparisons, yielding 100% recovery with maximum score error $9.54\times10^{-6}$. A strict permutation alignment fitted on 2,000 paired, unlabeled training activations recovered the planted mapping exactly in all four seed combinations. It achieved zero validation relative RMSE, cosine similarity 1.0, and exact score recovery in all 12 comparisons. The complete secondary-metric contract and 101,940 row-level predictions are retained in the public artifacts.

## Interpretation and next step

This experiment extends the causal Pythia control to a modern RMSNorm/SwiGLU transformer: coordinate mismatch alone is sufficient to destroy linear and nonlinear probe transfer, and label-free activation matching can recover a known symmetry exactly. It does not establish that natural cross-family gaps are exact parameter symmetries. The next experiment should test additional exact symmetry classes—MLP-neuron permutations, attention-component permutations, and valid positive rescalings—before comparing their signatures with the residual failures observed between independently trained model families.
