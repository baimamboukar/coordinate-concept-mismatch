from pathlib import Path

import pytest
import torch

from probe_transfer.extraction.activations import save_activation_file
from probe_transfer.transfer.evaluation import _pair_group
from probe_transfer.transfer.runner import _validate_activations


def test_assigns_explicit_transfer_pair_groups() -> None:
    config = {
        "evaluation": {
            "pair_groups": {
                "primary": [["llama", "qwen"], ["qwen", "llama"]],
                "lineage_control": [["llama", "nemotron"]],
            }
        }
    }

    assert _pair_group(config, "qwen", "llama") == "primary"
    assert _pair_group(config, "llama", "nemotron") == "lineage_control"
    with pytest.raises(ValueError, match="not assigned"):
        _pair_group(config, "qwen", "nemotron")


def test_requires_every_staged_model_split(tmp_path: Path) -> None:
    names = ("test", "seed_42_train", "seed_42_validation")
    config = {
        "models": {
            name: {
                "id": f"fake/{name}",
                "revision": "a" * 40,
                "layers": 4,
                "hidden_size": 2,
            }
            for name in ("llama", "qwen")
        },
        "data_seeds": [42],
        "dataset": {"id": "fake/data", "revision": "b" * 40},
        "sampling": {"test_size": 2, "train_size": 2, "validation_size": 2},
        "activations": {"normalized_depths": [0.75]},
    }
    for model in config["models"]:
        directory = tmp_path / "activations" / model
        directory.mkdir(parents=True)
        for split in names:
            save_activation_file(
                directory / f"{split}.safetensors",
                {
                    "layer_75": torch.zeros((2, 2)),
                    "row_ids": torch.arange(2),
                    "labels": torch.tensor([0, 1]),
                },
                {
                    "model": config["models"][model]["id"],
                    "model_revision": "a" * 40,
                    "dataset": "fake/data",
                    "dataset_revision": "b" * 40,
                    "split": split,
                    "rows": "2",
                },
            )

    _validate_activations(tmp_path, config)
    (tmp_path / "activations" / "qwen" / "test.safetensors").unlink()
    with pytest.raises(FileNotFoundError, match="qwen/test"):
        _validate_activations(tmp_path, config)
