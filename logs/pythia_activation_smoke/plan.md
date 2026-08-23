# Pythia activation smoke test

Extract last-non-padding residual-stream activations from 100 balanced WildGuardMix
training prompts using the pinned Pythia-410M checkpoint. Capture four normalized depths,
verify repeatability on eight rows, reject excessive truncation, and upload the verified
Safetensors artifact to the project Hugging Face bucket.

This is an engineering gate for the extraction pipeline, not evidence that probes transfer
or fail to transfer between modern model families.
