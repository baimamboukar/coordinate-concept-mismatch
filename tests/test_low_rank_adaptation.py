import numpy as np
import pytest
import torch

from core.config import ConfigError, load_config
from core.constants import CONFIGS_DIR
from pipeline.adaptation_batch import _validate_panel
from pipeline.config import materialize_stage
from pipeline.panel import select_task
from probe_transfer.alignment.fit import fit_evaluation_maps
from probe_transfer.alignment.methods import AlignmentMap
from probe_transfer.alignment.task_adaptation import (
    fit_task_adaptations,
    method_metadata,
    validate_task_adaptation,
)


def _settings() -> dict:
    return {
        "base_method": "affine_ridge",
        "calibration_split": "calibration",
        "calibration_rows": [64, 128],
        "ranks": [1, 2],
        "confirmatory_rows": 64,
        "confirmatory_rank": 2,
        "relative_alpha": 1e-6,
        "reference_method": "affine_ridge",
    }


def test_low_rank_correction_recovers_a_rank_two_task_residual() -> None:
    rng = np.random.default_rng(42)
    target = rng.normal(size=(128, 6)).astype(np.float32)
    update = (rng.normal(size=(6, 2)) @ rng.normal(size=(2, 6)) * 0.2).astype(np.float32)
    source = target @ (np.eye(6, dtype=np.float32) + update)
    base = AlignmentMap("affine_ridge", weight=torch.eye(6), bias=torch.zeros(6))

    maps = fit_task_adaptations(base, source, target, _settings(), shuffle_seed=314, device="cpu")
    rank_one = maps["low_rank_r1_n128"].transform(target)
    rank_two = maps["low_rank_r2_n128"].transform(target)
    shuffled = maps["shuffled_low_rank_r2_n128"].transform(target)

    assert np.mean((rank_two - source) ** 2) < np.mean((rank_one - source) ** 2)
    assert np.mean((rank_two - source) ** 2) < 1e-8
    assert np.mean((shuffled - source) ** 2) > 0.01
    weight = maps["low_rank_r2_n128"].weight
    assert weight is not None
    correction = weight - torch.eye(6)
    assert torch.linalg.matrix_rank(correction, atol=1e-4) <= 2


def test_evaluation_fit_composes_shared_and_task_specific_maps(tmp_path, monkeypatch) -> None:
    rng = np.random.default_rng(137)
    target = rng.normal(size=(128, 4)).astype(np.float32)
    source = target @ np.diag([1.5, 1.0, 1.0, 1.0]).astype(np.float32)
    paired = (source, target, np.arange(128), np.zeros(128))
    base = AlignmentMap("affine_ridge", weight=torch.eye(4), bias=torch.zeros(4))
    config = {
        "data_seeds": [42, 137],
        "models": {"first": {"hidden_size": 4}, "second": {"hidden_size": 4}},
        "materials": {"expected_calibration_rows": 128},
        "alignment": {
            "fit_split": "calibration",
            "shuffled_pairing_seed": 314,
            "task_adaptation": _settings(),
        },
    }
    monkeypatch.setattr(
        "probe_transfer.alignment.fit.cross_task.load_fit_split", lambda *_args: paired
    )
    monkeypatch.setattr(
        "probe_transfer.alignment.fit.cross_task.fit_expected_rows", lambda *_args: 128
    )
    monkeypatch.setattr("probe_transfer.alignment.fit.paired_split", lambda *_args: paired)
    monkeypatch.setattr(
        "probe_transfer.alignment.fit.fit_configured_alignments",
        lambda *_args, **_kwargs: ({"affine_ridge": base}, [{"selected": True}]),
    )

    maps, records, train = fit_evaluation_maps(
        tmp_path,
        tmp_path,
        config,
        source="first",
        target="second",
        data_seed=42,
        depth=0.75,
        layer="layer_75",
        device="cpu",
    )

    assert len(maps) == 9
    assert records[0]["source_model"] == "first"
    assert np.array_equal(train[0], source)


@pytest.mark.parametrize(
    "study_name",
    ["smollm_task_specific_low_rank_correction", "olmo1_task_specific_low_rank_correction"],
)
def test_task_adaptation_protocol_is_identical_across_model_pairs(study_name: str) -> None:
    study = load_config(CONFIGS_DIR / "studies" / f"{study_name}.yaml")
    conditions, tasks = _validate_panel(study)
    assert conditions == ["reconstruction_selected", "probe_selected"]
    assert tasks == ["ag_news", "mnli"]

    config = materialize_stage(select_task(study, "ag_news", "probe_selected"), "align")
    assert config["alignment"]["primary_method"] == "low_rank_r8_n256"
    assert config["alignment"]["negative_control"] == "shuffled_low_rank_r8_n256"
    assert config["expected_outputs"] == {
        "metrics_rows": 208,
        "prediction_rows": 416000,
        "recovery_rows": 200,
        "alignment_diagnostic_rows": 200,
        "alignment_selection_rows": 560,
    }


def test_invalid_confirmatory_endpoint_is_rejected() -> None:
    study = load_config(CONFIGS_DIR / "studies" / "smollm_task_specific_low_rank_correction.yaml")
    config = materialize_stage(select_task(study, "mnli", "probe_selected"), "align")
    config["alignment"]["task_adaptation"]["confirmatory_rows"] = 512
    with pytest.raises(ConfigError, match="row budgets"):
        validate_task_adaptation(config)


def test_low_rank_method_metadata_is_structured() -> None:
    assert method_metadata("low_rank_r8_n256") == {
        "map_class": "task_specific_low_rank",
        "correction_rank": 8,
        "calibration_rows": 256,
    }
    assert method_metadata("affine_ridge") == {}
