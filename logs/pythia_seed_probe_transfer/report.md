# Pythia Seed Probe Transfer

Status: implementation validated; remote run pending.

## Prepared data

- Dataset: `allenai/wildguardmix` at
  `d29c47f41c8b51348b5c8e8c81c039b3132b66d1`.
- Seed 42: 12,000 balanced train rows and 2,000 balanced validation rows.
- Seed 137: 12,000 balanced train rows and 2,000 balanced validation rows.
- Protected test: 1,699 rows (945 unharmful and 754 harmful).
- Cleaning removed 38,924 duplicate train rows, 15 conflicting-label prompts,
  14 invalid train prompts, 3 train-test overlaps, and 26 unlabeled test rows.
- Prepared payload: 5,765,204 bytes; SHA-256
  `e1d2c873cd30214fbb99ee66276b729fc7aa45ab7804a940a5c38f66c9e62b11`.

Both pinned checkpoint tokenizers produced the same tokenization digest,
`e3be0dd47896e904f9cdab12ee5a7d75c49736ea139a027d8cfdbe603b6d661b`.
At the prespecified 512-token limit, truncation ranges from 0.18% on the protected
test set to 0.81% on the seed-42 training split, below the 5% gate.

## Local validation

Ruff, formatting, Pyright, and 39 tests pass. The tests include an end-to-end
synthetic coordinate reversal in which both oracle AUROCs are 1.0 and both frozen
cross-model probes fail with an AUROC gap of 1.0.
