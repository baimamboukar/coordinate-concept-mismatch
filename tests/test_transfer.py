from pathlib import Path

import torch

from core.tracking import Tracker
from probe_transfer.activations import save_activation_file
from probe_transfer.transfer import _bootstrap_seed, run_transfer


def _write_split(path: Path, values: list[float], labels: list[int]) -> None:
    activations = torch.tensor([[value, value / 2] for value in values], dtype=torch.float16)
    save_activation_file(
        path,
        {
            "layer_75": activations,
            "row_ids": torch.arange(len(values)),
            "labels": torch.tensor(labels),
        },
        {"split": path.stem},
    )


def test_transfer_pipeline_detects_synthetic_coordinate_failure(tmp_path: Path) -> None:
    labels = [0, 0, 1, 1]
    values = [-2.0, -1.0, 1.0, 2.0]
    for model, sign in (("model_a", 1), ("model_b", -1)):
        directory = tmp_path / "activations" / model
        _write_split(directory / "seed_42_train.safetensors", [sign * x for x in values], labels)
        _write_split(
            directory / "seed_42_validation.safetensors", [sign * x for x in values], labels
        )
        _write_split(directory / "test.safetensors", [sign * x for x in values], labels)

    config = {
        "seed": 42,
        "data_seeds": [42],
        "models": {"model_a": {}, "model_b": {}},
        "activations": {"normalized_depths": [0.75], "primary_depth": 0.75},
        "probes": {
            "primary_families": ["linear"],
            "secondary_families": ["linear"],
            "linear": {"c_values": [0.1, 1.0], "max_iter": 100},
        },
        "evaluation": {
            "operating_fprs": [0.01, 0.05],
            "bootstrap_samples": 20,
            "confidence_level": 0.95,
            "oracle_gate": 0.75,
            "minimum_gap": 0.10,
        },
    }
    tracker = Tracker("synthetic_transfer", tmp_path / "report.md")

    gaps, checksums = run_transfer(tmp_path, config, tracker)

    assert len(gaps) == 2
    assert all(gap["transfer_failed"] for gap in gaps)
    assert all(gap["auroc_gap"] == 1.0 for gap in gaps)
    assert "results/metrics.jsonl" in checksums
    assert (tmp_path / "results" / "predictions.jsonl").is_file()


def test_bootstrap_seeds_are_unique_for_five_model_directions() -> None:
    models = ["llama", "qwen", "nemotron", "mistral", "granite"]
    seeds = {
        _bootstrap_seed(42, (137, 0.75, "linear", source, target), models)
        for source in models
        for target in models
        if source != target
    }

    assert len(seeds) == 20
