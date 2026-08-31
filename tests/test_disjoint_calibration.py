import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from core.config import ConfigError, load_config
from core.constants import CONFIGS_DIR
from core.tracking import Tracker
from pipeline.config import materialize_stage
from pipeline.panel import select_task, task_variants
from probe_transfer.alignment.contrasts import condition_contrasts
from probe_transfer.alignment.evaluation import evaluate_checkpoint_alignment
from probe_transfer.alignment.runner import _assert_expected_outputs
from probe_transfer.data import assert_disjoint_prepared_splits, prepare_splits
from probe_transfer.extraction.activations import save_activation_file
from probe_transfer.extraction.job import prepared_splits
from probe_transfer.splits import validate_split_configuration
from probe_transfer.transfer.evaluation import run_transfer


def _parameters():
    return {
        "train_size": 20,
        "validation_size": 8,
        "seeds": [42, 137],
        "prompt_field": "text",
        "label_field": "label",
        "positive_label": 1,
        "negative_label": 0,
        "adversarial_field": None,
    }


def _calibration():
    return {
        "train_size": 24,
        "validation_size": 8,
        "holdout_seed": 314,
        "evaluation_source": "unused_training_pool",
    }


def _rows():
    return [
        {"text": f"example {label} {i}", "label": label} for label in (0, 1) for i in range(120)
    ]


def test_fresh_partitions_preserve_prior_splits_and_exclude_both_seeds() -> None:
    official = [{"text": "example 0 0", "label": 0}, {"text": "test", "label": 1}]
    old_test, original, _ = prepare_splits(_rows(), official, **_parameters())
    arguments = {**_parameters(), "calibration": _calibration(), "fresh_test_size": 20}
    test, seeded, audit = prepare_splits(_rows(), official, **arguments)
    assert (test, seeded, audit) == prepare_splits(_rows(), official, **arguments)
    used = {
        row["prompt_sha256"]
        for rows in original.values()
        for split in rows.values()
        for row in split
    }
    used.update(row["prompt_sha256"] for row in old_test)
    assert not used & {row["prompt_sha256"] for row in test}
    flat = {"test": test}
    for seed, splits in seeded.items():
        assert splits["train"] == original[seed]["train"]
        assert splits["validation"] == original[seed]["validation"]
        for name, rows in splits.items():
            flat[f"seed_{seed}_{name}"] = rows
            assert 2 * sum(row["label"] for row in rows) == len(rows)
            if "calibration" in name:
                assert not used & {row["prompt_sha256"] for row in rows}
    assert_disjoint_prepared_splits(flat)
    assert audit["partition"]["fresh_test_rows"] == 20
    flat["seed_42_calibration"][0]["prompt"] = flat["seed_137_train"][0]["prompt"].upper()
    with pytest.raises(ValueError, match="overlap prior probe"):
        assert_disjoint_prepared_splits(flat)


def test_insufficient_calibration_pool_is_not_silently_resampled() -> None:
    with pytest.raises(ValueError, match="Insufficient unused rows"):
        prepare_splits(
            _rows(),
            [],
            **_parameters(),
            fresh_test_size=20,
            calibration={**_calibration(), "train_size": 300},
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("train_size", 3),
        ("validation_size", True),
        ("holdout_seed", -1),
        ("evaluation_source", "test"),
    ],
)
def test_invalid_disjoint_settings_fail_before_execution(field, value) -> None:
    config = {
        "sampling": {
            "train_size": 20,
            "validation_size": 8,
            "test_size": 20,
            "disjoint_calibration": {**_calibration(), field: value},
        }
    }
    with pytest.raises(ConfigError):
        validate_split_configuration(config)


def test_configuration_isolates_fit_roles_and_disables_previous_tests() -> None:
    study = load_config(CONFIGS_DIR / "studies/smollm_disjoint_calibration.yaml")
    assert len(list(task_variants(study, "align"))) == 10
    assert {row["task"] for row in task_variants(study, "extract")} == {"sst2", "wildguard"}
    assert len(prepared_splits(select_task(study, "sst2"))) == 9
    for condition, split in (("overlap_uniform", "train"), ("disjoint_uniform", "calibration")):
        config = materialize_stage(select_task(study, "sst2", condition), "align")
        assert config["alignment"]["fit_split"] == split
        assert config["alignment"]["diagnostic_split"] == "calibration_validation"
        assert config["materials"]["expected_calibration_rows"] == 12000
        assert config["materials"]["expected_test_rows"] == 2000
        assert config["expected_outputs"]["alignment_selection_rows"] == 16
        assert config["reference_materials"]["source_study"] == study["name"]
    with pytest.raises(ConfigError, match="Select --task"):
        select_task(study, "ag_news")
    bad = copy.deepcopy(study)
    bad["fit_conditions"]["disjoint_uniform"]["split"] = "test"
    with pytest.raises(ConfigError, match="configured seeded split"):
        select_task(bad, "sst2", "disjoint_uniform")


def test_disjoint_alignment_and_paired_contrasts_end_to_end(tmp_path: Path) -> None:
    study = load_config(CONFIGS_DIR / "studies/smollm_disjoint_calibration.yaml")
    study["sampling"].update(train_size=64, validation_size=24)
    study["sampling"]["disjoint_calibration"].update(train_size=96, validation_size=24)
    study["probes"]["primary_families"] = ["linear"]
    study["evaluation"]["bootstrap_samples"] = 8
    study["pipeline"]["stages"]["align"]["alignment"]["device"] = "cpu"
    for spec in study["tasks"].values():
        if spec is not None:
            spec["sampling"]["test_size"] = 40
    for model in study["models"].values():
        model["hidden_size"] = 3
    for spec in study["fit_conditions"].values():
        if spec is not None:
            size = 64 if spec["split"] == "train" else 96
            spec["datasets"] = {"sst2": size, "wildguard": size}
    sources, references = {}, {}
    fit = tmp_path / "fit"
    rng = np.random.default_rng(18)
    for task in ("sst2", "wildguard"):
        config = materialize_stage(select_task(study, task), "transfer")
        source = tmp_path / f"baseline-{task}"
        for split in prepared_splits(config):
            target = rng.normal(size=(split.expected_rows, 3)).astype(np.float32)
            values = -target * (2 if split.name.endswith("_calibration") else 1)
            labels = torch.from_numpy((values[:, 0] > 0).astype(np.int64))
            for model, array in (("smollm1", values), ("smollm2", target)):
                save_activation_file(
                    source / "activations" / model / split.output_name,
                    {
                        "layer_75": torch.from_numpy(array),
                        "row_ids": torch.arange(len(values)),
                        "labels": labels,
                    },
                    {},
                )
        run_transfer(source, config, Tracker("synthetic", None))
        sources[task] = source
        entry = fit / config["artifacts"]["dataset_key"]
        entry.mkdir(parents=True)
        (entry / "activations").symlink_to(source / "activations", target_is_directory=True)
        reference = tmp_path / f"reference-{task}"
        evaluate_checkpoint_alignment(
            source, reference, materialize_stage(select_task(study, task), "align")
        )
        references[task] = reference / "results/recovery.jsonl"
    signatures = {}
    for task, baseline in sources.items():
        for name, spec in study["fit_conditions"].items():
            if spec is None:
                continue
            config = materialize_stage(select_task(study, task, name), "align")
            destination = tmp_path / task / name
            evaluate_checkpoint_alignment(
                baseline, destination, config, fit_root=fit, reference_path=references[task]
            )
            _assert_expected_outputs(destination, config)
            records = [
                json.loads(line)
                for line in (destination / "results/alignment_selection.jsonl")
                .read_text()
                .splitlines()
            ]
            assert {row["fit_rows"] for row in records} == {64 if spec["split"] == "train" else 96}
            signatures[task, name] = [row["map_fingerprint"] for row in records if row["selected"]]
    assert signatures["sst2", "disjoint_uniform"] != signatures["sst2", "overlap_uniform"]
    assert signatures["sst2", "disjoint_uniform"] == signatures["wildguard", "disjoint_uniform"]
    contrasts = condition_contrasts(tmp_path, study, ["sst2", "wildguard"])
    assert len(contrasts) == 16
    assert {row["probe_family"] for row in contrasts} == {"linear"}
    assert all(row["ci_lower"] <= row["ci_upper"] for row in contrasts)
