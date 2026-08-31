import json

import pytest
import torch
from safetensors.torch import save_file

from core.config import ConfigError, load_config
from core.constants import CONFIGS_DIR
from core.tracking import Tracker
from pipeline.config import materialize_stage
from pipeline.materials import prepare_panel_materials, validate_material_preparation
from pipeline.panel import select_task
from probe_transfer.materialization import verify_prior_probe_splits
from probe_transfer.preparation import prepare_dataset


def _study():
    return load_config(CONFIGS_DIR / "studies/smollm_disjoint_calibration.yaml")


def test_both_data_preparations_precede_every_inference(tmp_path, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr("pipeline.materials._run_stage", lambda *args: calls.append(args[3:]))
    monkeypatch.setattr("pipeline.materials.verify_prior_probe_splits", lambda *_args: 4)
    prepare_panel_materials(_study(), CONFIGS_DIR / "study.yaml", tmp_path, ["sst2", "wildguard"])
    assert calls[:2] == [("sst2", "prepare"), ("wildguard", "prepare")]
    assert [row[1] for row in calls[2:6]] == ["preflight"] * 4
    assert [row[1] for row in calls[6:10]] == ["extract"] * 4
    assert [row[1] for row in calls[10:]] == ["transfer", "align", "transfer", "align"]


def test_material_generation_cannot_overwrite_a_previous_study() -> None:
    study = _study()
    study["reuse_materials"]["study"] = "older_study"
    with pytest.raises(ConfigError, match="current study"):
        validate_material_preparation(study)
    study = _study()
    study["tasks"]["heldout"] = {"role": "held_out"}
    with pytest.raises(ConfigError, match="included-task-only"):
        validate_material_preparation(study)


def test_preparation_resume_verifies_the_exact_pinned_recipe(tmp_path, monkeypatch) -> None:
    config = materialize_stage(select_task(_study(), "sst2"), "prepare")
    config["sampling"].update(train_size=20, validation_size=8, test_size=20)
    config["sampling"]["disjoint_calibration"].update(train_size=24, validation_size=8)
    data = [
        {"sentence": f"example {label} {index}", "label": label}
        for label in (0, 1)
        for index in range(150)
    ]
    monkeypatch.setenv("ACTIVATION_ROWS_DIR", str(tmp_path / "rows"))
    monkeypatch.setattr(
        "probe_transfer.preparation.load_huggingface_dataset",
        lambda *_args, split, **_kwargs: data if split == "train" else data[:2],
    )
    root = prepare_dataset(config, Tracker("data", None))
    assert prepare_dataset(config, Tracker("data", None)) == root
    audit = json.loads((root / "split_audit.json").read_text())
    assert audit["partition"]["overlap_rows"] == 0
    path = root / "seed_42_calibration.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["label"] = 1 - rows[0]["label"]
    path.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError, match="pinned recipe"):
        prepare_dataset(config, Tracker("data", None))


def test_prior_probe_rows_are_checked_against_published_activations(tmp_path, monkeypatch) -> None:
    config = materialize_stage(select_task(_study(), "sst2"), "prepare")
    config["sampling"].update(train_size=2, validation_size=2)
    monkeypatch.setattr(
        "probe_transfer.materialization._sync_public", lambda *_args, **_kwargs: None
    )
    rows_root, cache = tmp_path / "rows", tmp_path / "cache"
    rows_root.mkdir()
    cache.mkdir()
    for seed in config["data_seeds"]:
        for split in ("train", "validation"):
            name = f"seed_{seed}_{split}"
            rows = [{"row_id": i, "prompt": f"prompt {i}", "label": i} for i in range(2)]
            (rows_root / f"{name}.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
            model = config["models"]["smollm1"]
            metadata = {
                "model": model["id"],
                "model_revision": model["revision"],
                "dataset": config["dataset"]["id"],
                "dataset_revision": config["dataset"]["revision"],
                "split": name,
                "rows": "2",
            }
            save_file(
                {"row_ids": torch.arange(2), "labels": torch.arange(2)},
                cache / f"{name}.safetensors",
                metadata=metadata,
            )
    assert verify_prior_probe_splits(config, rows_root, cache) == 4
    path = rows_root / "seed_42_train.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["row_id"] = 99
    path.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError, match="row_id order differs"):
        verify_prior_probe_splits(config, rows_root, cache)
