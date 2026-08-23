# Pythia activation smoke test

Status: completed on 2026-08-23.

## Setup

- Code: `3fb0073e20df90ff1976e5b0e9f69fcd21936aef`.
- Model: `EleutherAI/pythia-410m` at
  `9879c9b5f8bea9051dcb0e68dff21493d67e9d4f`.
- Data: 100 balanced WildGuardMix rows from seed 42; input SHA-256
  `cb639d71b2c48bc0ece40d26edf4af999f9577fb13aacd563d0319526027c2a2`.
- Activations: last non-padding token, FP16, maximum length 512, normalized depths
  25%, 50%, 75%, and 100% (blocks 6, 12, 18, and 24).
- Compute: one Vast.ai H100 PCIe, instance `48492446`, label `823410`.
  The worker received no Hugging Face credentials and was destroyed after retrieval.

## Results

| Metric | Role | Result |
| --- | --- | --- |
| Hugging Face round-trip checksum | Primary | Pass |
| Eight-row repeatability at absolute tolerance `1e-4` | Primary | Pass |
| Truncation rate | Secondary | 0.00% |
| Extracted rows | Secondary | 100 |
| Label balance | Secondary | 50 unharmful / 50 harmful |
| Residual-stream shape per depth | Secondary | `[100, 1024]` |

Artifact:
`hf://buckets/baimamboukar/coordinate-concept-mismatch/experiments/pythia_activation_smoke/activations/smoke/seed_42/pythia_410m.safetensors`

- Size: 821,844 bytes.
- SHA-256: `3668539d78d56fdab3fb31d66fb930b13aee3ecb4deafac69ab89639c60f4335`.
- Stored tensors: four activation matrices, row IDs, labels, and adversarial flags.

## Interpretation

The extraction, repeatability, serialization, secure transfer, and durable upload pipeline
works for Pythia-410M. No probe was trained in this smoke test, so AUROC, AUPRC, accuracy,
balanced accuracy, precision, recall, F1, calibration, confusion counts, and low-FPR metrics
were not computed. Pythia remains an engineering control, not paper evidence about modern
cross-model probe transfer.
