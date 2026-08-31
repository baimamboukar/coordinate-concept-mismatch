# Coordinate or Concept Mismatch?

Research code for disentangling coordinate mismatch from representational differences in cross-model probe transfer.

The repository has one configuration-driven pipeline. A study YAML selects reusable stages—data preparation, activation preflight and extraction, frozen-probe transfer, activation alignment, or exact-symmetry controls. Adding a model pair or ablation changes configuration; it does not require a new Python runner.

Multi-task panels share one YAML: `--task` selects the evaluation dataset, and `--fit` selects an alignment condition.

```bash
PYTHONHASHSEED=42 uv run python src/run.py \
  configs/studies/modern_models.yaml transfer --validate-only

PYTHONHASHSEED=42 uv run python src/run.py \
  configs/studies/modern_models.yaml extract --model mistral

PYTHONHASHSEED=42 uv run python src/run.py \
  configs/studies/smollm_heldout_map_replication.yaml align \
  --task ag_news --fit pooled_equal --validate-only
```

`src/pipeline/` owns orchestration, while `src/probe_transfer/` contains reusable extraction, probe, transfer, alignment, and symmetry components. Prepared prompt splits are rebuilt deterministically from the pinned dataset and remain worker-local. Workers read shared inputs from and publish activations, probes, and results directly to the [public Hugging Face bucket](https://huggingface.co/buckets/baimamboukar/coordinate-concept-mismatch); training is tracked with [Weights & Biases](https://wandb.ai/JinesisLab/coordinate-concept-mismatch).
