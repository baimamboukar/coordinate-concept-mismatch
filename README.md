# Coordinate or Concept Mismatch?

This research project studies why probes trained on one language model's internal activations often fail when applied unchanged to another model. We ask whether transfer failure comes from different internal coordinate systems or from genuinely different learned representations.

The first experiment measures frozen probe transfer across Llama 3.1, Mistral, Qwen3, and Nemotron using harmfulness labels from WildGuardMix. Later experiments will introduce function-preserving parameter symmetries and representation-alignment methods to separate coordinate mismatch from concept mismatch.

The repository contains reproducible YAML configurations, activation extraction, probe training and evaluation components, experiment plans, and complete operating metrics. Large activation and result artifacts are stored in the project's public Hugging Face bucket.

## Development

```bash
uv sync
uv run pre-commit install --install-hooks
uv run pytest
```

GPU experiments are launched only after their configuration, expected cost, and artifact destination have been reviewed.
