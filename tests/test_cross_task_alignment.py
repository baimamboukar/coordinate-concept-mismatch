import json
from pathlib import Path

import numpy as np
import pytest

from probe_transfer.alignment.cross_task import (
    add_improvement_retention,
    load_fit_split,
    load_recovery_reference,
    validate_cross_task_alignment,
)


def _row(**updates):
    row = {
        "data_seed": 42,
        "depth": 0.75,
        "probe_family": "linear",
        "source_model": "ai2",
        "target_model": "amd",
        "pair_group": "primary",
        "method": "affine_ridge",
        "raw_auroc_gap": 0.4,
        "aligned_auroc": 0.8,
        "aligned_auroc_improvement": 0.3,
        "recovery_fraction": 0.75,
    }
    return {**row, **updates}


def test_cross_task_recovery_records_same_task_retention(tmp_path: Path) -> None:
    path = tmp_path / "recovery.jsonl"
    path.write_text(json.dumps(_row()) + "\n")
    reference = load_recovery_reference(path)

    enriched = add_improvement_retention(
        _row(aligned_auroc=0.7, aligned_auroc_improvement=0.225, recovery_fraction=0.5625),
        reference,
        tolerance=1e-5,
    )

    assert enriched["same_task_aligned_auroc"] == 0.8
    assert enriched["same_task_recovery_fraction"] == 0.75
    assert enriched["improvement_retention"] == pytest.approx(0.75)


def test_cross_task_recovery_rejects_a_different_baseline(tmp_path: Path) -> None:
    path = tmp_path / "recovery.jsonl"
    path.write_text(json.dumps(_row()) + "\n")

    with pytest.raises(ValueError, match="baselines differ"):
        add_improvement_retention(
            _row(raw_auroc_gap=0.2),
            load_recovery_reference(path),
            tolerance=1e-5,
        )


def test_pooled_fit_concatenates_equal_task_budgets(tmp_path: Path, monkeypatch) -> None:
    config = {
        "materials": {"expected_train_rows": 4},
        "fit_materials": {
            "expected_train_rows": 4,
            "task_balanced": True,
            "datasets": [
                {"dataset_key": "first-v1", "expected_train_rows": 6, "fit_rows": 2},
                {"dataset_key": "second-v1", "expected_train_rows": 6, "fit_rows": 2},
            ],
        },
    }

    def fake_split(root, *_args):
        offset = 0 if root.name == "first-v1" else 10
        values = np.arange(6) + offset
        return values[:, None], values[:, None], values, values % 2

    monkeypatch.setattr("probe_transfer.alignment.cross_task.paired_split", fake_split)
    values = load_fit_split(tmp_path, config, "source", "target", "seed_42_train", "layer_75")

    assert values[0][:, 0].tolist() == [0, 1, 10, 11]
    assert all(len(item) == 4 for item in values)


def test_task_balanced_fit_rejects_unequal_rows() -> None:
    config = {
        "artifacts": {"dataset_key": "boolq-v1"},
        "fit_materials": {
            "expected_train_rows": 6,
            "task_balanced": True,
            "datasets": [
                {
                    "dataset_key": "sst2-v1",
                    "source_study": "sst2",
                    "expected_train_rows": 10,
                    "fit_rows": 2,
                },
                {
                    "dataset_key": "wildguard-v1",
                    "source_study": "wildguard",
                    "expected_train_rows": 10,
                    "fit_rows": 4,
                },
            ],
        },
        "reference_materials": {
            "source_name": "alignment",
            "source_study": "boolq",
            "source_variant": "same-task",
        },
    }

    with pytest.raises(ValueError, match="equal rows"):
        validate_cross_task_alignment(config)


def _fit_entry(dataset_key: str) -> dict:
    return {
        "dataset_key": dataset_key,
        "source_study": dataset_key,
        "expected_train_rows": 12,
        "fit_rows": 6,
    }


def _pooled_config(**fit_updates) -> dict:
    fit = {
        "expected_train_rows": 12,
        "task_balanced": True,
        "datasets": [_fit_entry("sst2-v1"), _fit_entry("wildguard-v1")],
        **fit_updates,
    }
    return {
        "artifacts": {"dataset_key": "sst2-v1"},
        "fit_materials": fit,
        "reference_materials": {
            "source_name": "alignment",
            "source_study": "sst2",
            "source_variant": "same-task",
        },
    }


def test_pooled_fit_may_include_evaluation_training_rows() -> None:
    config = _pooled_config(evaluation_included=True)

    validate_cross_task_alignment(config)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({}, "only when evaluation_included is true"),
        (
            {"evaluation_included": True, "datasets": [_fit_entry("wildguard-v1")]},
            "requires the evaluation dataset",
        ),
        (
            {"evaluation_included": True, "datasets": [_fit_entry("sst2-v1")]},
            "and a distinct fit dataset",
        ),
    ],
)
def test_pooled_fit_requires_an_explicit_distinct_evaluation_mix(updates, message) -> None:
    config = _pooled_config()
    config["fit_materials"].update(updates)
    config["fit_materials"]["expected_train_rows"] = sum(
        entry["fit_rows"] for entry in config["fit_materials"]["datasets"]
    )

    with pytest.raises(ValueError, match=message):
        validate_cross_task_alignment(config)
