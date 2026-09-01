import json
from pathlib import Path

import numpy as np
import pytest
import torch

from core.config import load_config
from core.constants import CONFIGS_DIR, REQUIRED_ROW_LEVEL_FIELDS
from core.tracking import Tracker
from pipeline.batch import _assert_shared_maps, eligible_conditions, run_alignment_panel
from pipeline.config import materialize_stage
from pipeline.panel import select_task
from probe_transfer.alignment.evaluation import evaluate_checkpoint_alignment
from probe_transfer.alignment.runner import _assert_expected_outputs
from probe_transfer.extraction.activations import save_activation_file
from probe_transfer.transfer.evaluation import run_transfer


def _tiny_study():
    study = load_config(CONFIGS_DIR / "studies/smollm_shared_map_compatibility.yaml")
    study["sampling"].update(train_size=64, validation_size=24)
    study["probes"]["primary_families"] = ["linear"]
    study["probes"]["linear"]["c_values"] = [1.0]
    study["evaluation"]["bootstrap_samples"] = 8
    study["pipeline"]["stages"]["align"]["alignment"]["device"] = "cpu"
    for spec in study["tasks"].values():
        spec.setdefault("sampling", {})["test_size"] = 40
    for model in study["models"].values():
        model["hidden_size"] = 3
    for condition in study["fit_conditions"].values():
        if condition is not None:
            condition["datasets"] = {"sst2": 64, "wildguard": 64}
    return study


def _activations(root: Path, config, scale) -> None:
    rng = np.random.default_rng(42)
    splits = {
        "test": 40,
        **{
            f"seed_{seed}_{split}": size
            for seed in config["data_seeds"]
            for split, size in (("train", 64), ("validation", 24))
        },
    }
    for split, size in splits.items():
        target = rng.normal(size=(size, 3)).astype(np.float32) * scale
        source = -target
        labels = torch.from_numpy((source[:, 0] > 0).astype(np.int64))
        for model, values in (("smollm1", source), ("smollm2", target)):
            save_activation_file(
                root / "activations" / model / f"{split}.safetensors",
                {
                    "layer_75": torch.from_numpy(values),
                    "row_ids": torch.arange(size),
                    "labels": labels,
                },
                {},
            )


def test_cached_alignment_end_to_end_keeps_maps_task_independent(tmp_path: Path) -> None:
    study = _tiny_study()
    fit_root = tmp_path / "fit"
    source_roots, references = {}, {}
    for task, scale in (("sst2", 1), ("wildguard", 3)):
        config = materialize_stage(select_task(study, task), "transfer")
        root = tmp_path / task
        _activations(root, config, scale)
        run_transfer(root, config, Tracker("synthetic", None))
        source_roots[task] = root
        dataset_root = fit_root / config["artifacts"]["dataset_key"]
        dataset_root.mkdir(parents=True)
        (dataset_root / "activations").symlink_to(root / "activations", target_is_directory=True)
        alignment = materialize_stage(select_task(study, task), "align")
        reference = tmp_path / f"reference-{task}"
        evaluate_checkpoint_alignment(root, reference, alignment)
        references[task] = reference / "results/recovery.jsonl"
    fingerprints = []
    for task in ("sst2", "wildguard"):
        config = materialize_stage(select_task(study, task, "scale_balanced_selected"), "align")
        output = tmp_path / f"output-{task}"
        recoveries, _, checksums = evaluate_checkpoint_alignment(
            source_roots[task], output, config, fit_root=fit_root, reference_path=references[task]
        )
        _assert_expected_outputs(output, config)
        assert "results/alignment_selection.jsonl" in checksums
        assert all("improvement_retention" in row for row in recoveries)
        selection = [
            json.loads(line)
            for line in (output / "results/alignment_selection.jsonl").read_text().splitlines()
        ]
        fingerprints.append(
            {
                (row["data_seed"], row["source_model"], row["method"]): row["map_fingerprint"]
                for row in selection
                if row["selected"]
            }
        )
        predictions = [
            json.loads(line)
            for line in (output / "results/predictions.jsonl").read_text().splitlines()
        ]
        assert all(REQUIRED_ROW_LEVEL_FIELDS <= set(row) for row in predictions)
    assert fingerprints[0] == fingerprints[1]


def test_heldout_gate_requires_every_included_task_and_rejects_duplicates() -> None:
    values = [
        {"task": "sst2", "condition": "uniform", "passes_criterion": True},
        {"task": "wildguard", "condition": "uniform", "passes_criterion": False},
    ]
    assert eligible_conditions(values, ["sst2", "wildguard"], ["uniform"]) == []
    values[1]["passes_criterion"] = True
    assert eligible_conditions(values, ["sst2", "wildguard"], ["uniform"]) == ["uniform"]
    assert eligible_conditions(values + [values[0]], ["sst2", "wildguard"], ["uniform"]) == []


def test_batch_refuses_evaluation_specific_maps() -> None:
    with pytest.raises(ValueError, match="changed with the evaluation task"):
        _assert_shared_maps(
            [
                {"condition": "uniform", "map_fingerprints": {"map": "a"}},
                {"condition": "uniform", "map_fingerprints": {"map": "b"}},
            ]
        )


def test_batch_skips_heldout_compute_when_compatibility_fails(tmp_path: Path, monkeypatch) -> None:
    study = _tiny_study()
    calls, published = [], []
    monkeypatch.setenv("EXPERIMENT_OUTPUT_DIR", str(tmp_path))

    def run_task(_study, _path, _root, task, conditions):
        calls.extend((task, condition) for condition in conditions)
        return [
            {
                "task": task,
                "condition": condition,
                "passes_criterion": False,
                "map_fingerprints": {"shared": condition},
            }
            for condition in conditions
        ]

    monkeypatch.setattr("pipeline.batch._run_task", run_task)
    monkeypatch.setattr("pipeline.batch.publish_artifacts", lambda *_args: published.append(True))
    run_alignment_panel(study, CONFIGS_DIR / "studies/smollm_shared_map_compatibility.yaml")
    assert len(calls) == 8
    assert {task for task, _ in calls} == {"sst2", "wildguard"}
    summary = json.loads((tmp_path / "results/summary.json").read_text())
    assert summary["eligible_conditions"] == []
    assert len(summary["heldout_skipped_conditions"]) == 4
    assert published == [True]


def test_batch_prepares_heldout_materials_only_after_a_condition_qualifies(
    tmp_path: Path, monkeypatch
) -> None:
    study = _tiny_study()
    study["execution"]["prepare_materials"] = True
    calls = []
    monkeypatch.setenv("EXPERIMENT_OUTPUT_DIR", str(tmp_path))

    def prepare(_study, _path, _root, tasks):
        calls.append(("prepare", tuple(tasks)))

    def run_task(_study, _path, _root, task, conditions):
        calls.append(("evaluate", task, tuple(conditions)))
        return [
            {
                "task": task,
                "condition": condition,
                "passes_criterion": True,
                "map_fingerprints": {"shared": condition},
            }
            for condition in conditions
        ]

    monkeypatch.setattr("pipeline.batch.prepare_panel_materials", prepare)
    monkeypatch.setattr("pipeline.batch._run_task", run_task)
    monkeypatch.setattr("pipeline.batch.publish_artifacts", lambda *_args: None)
    run_alignment_panel(study, CONFIGS_DIR / "studies/smollm_shared_map_compatibility.yaml")

    heldout_prepare = calls.index(("prepare", ("ag_news", "mnli")))
    included_evaluations = [
        index
        for index, call in enumerate(calls)
        if call[0] == "evaluate" and call[1] in {"sst2", "wildguard"}
    ]
    heldout_evaluations = [
        index
        for index, call in enumerate(calls)
        if call[0] == "evaluate" and call[1] in {"ag_news", "mnli"}
    ]
    assert calls[0] == ("prepare", ("sst2", "wildguard"))
    assert max(included_evaluations) < heldout_prepare < min(heldout_evaluations)
