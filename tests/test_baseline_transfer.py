import json
from pathlib import Path

import pytest
import torch

from probe_transfer.activations import save_activation_file
from probe_transfer.baseline_transfer import _validate_activations
from probe_transfer.transfer import _pair_group


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
        "sampling": {"test_size": 2, "train_size": 2, "validation_size": 2},
        "activations": {"normalized_depths": [0.75]},
    }
    for model in config["models"]:
        directory = tmp_path / "activations" / model
        directory.mkdir(parents=True)
        completion = {
            "status": "complete",
            "model_name": model,
            "model_id": config["models"][model]["id"],
            "model_revision": "a" * 40,
            "block_indices": [3],
            "normalized_depths": [0.75],
            "splits": [{"split": split, "rows": 2} for split in names],
        }
        (directory / "completion.json").write_text(json.dumps(completion))
        for split in names:
            save_activation_file(
                directory / f"{split}.safetensors",
                {
                    "layer_75": torch.zeros((2, 2)),
                    "row_ids": torch.arange(2),
                    "labels": torch.tensor([0, 1]),
                },
                {},
            )

    _validate_activations(tmp_path, config)
    (tmp_path / "activations" / "qwen" / "test.safetensors").unlink()
    with pytest.raises(FileNotFoundError, match="qwen/test"):
        _validate_activations(tmp_path, config)
