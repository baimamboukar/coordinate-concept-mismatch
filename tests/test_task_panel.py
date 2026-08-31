import copy

import pytest

from core.config import ConfigError, load_config
from core.constants import CONFIGS_DIR, REQUIRED_BINARY_METRICS, REQUIRED_ROW_LEVEL_FIELDS
from pipeline.config import materialize_stage
from pipeline.panel import select_task, task_variants
from probe_transfer.layout import activation_prefix, stage_prefix


@pytest.fixture
def study():
    return load_config(CONFIGS_DIR / "studies" / "smollm_heldout_map_replication.yaml")


def test_panel_is_explicit_and_preserves_source_config(study) -> None:
    original = copy.deepcopy(study)
    selected = select_task(study, "mnli", "pooled_equal")

    assert study == original
    assert selected["task"] == "mnli"
    assert selected["task_role"] == "held_out"
    assert "tasks" not in selected
    with pytest.raises(ConfigError, match="Select --task"):
        select_task(study)
    with pytest.raises(ConfigError, match="Select a panel task"):
        materialize_stage(study, "transfer")


@pytest.mark.parametrize(
    "task,size", [("sst2", 872), ("wildguard", 1699), ("ag_news", 3800), ("mnli", 6692)]
)
def test_panel_reuses_protocol_and_derives_exact_contract(study, task, size) -> None:
    selected = select_task(study, task)
    transfer = materialize_stage(selected, "transfer")
    alignment = materialize_stage(selected, "align")

    assert list(selected["models"]) == ["smollm1", "smollm2"]
    assert selected["data_seeds"] == [42, 137]
    assert selected["activations"]["normalized_depths"] == [0.75]
    assert transfer["expected_outputs"]["metrics_rows"] == 24
    assert transfer["expected_outputs"]["prediction_rows"] == 24 * size
    assert transfer["expected_outputs"]["transfer_gap_rows"] == 12
    assert alignment["expected_outputs"]["metrics_rows"] == 60
    assert alignment["expected_outputs"]["recovery_rows"] == 36
    assert alignment["expected_outputs"]["prediction_rows"] == 60 * size
    assert alignment["materials"]["expected_test_rows"] == size
    assert alignment["materials"]["source_variant"] == transfer["artifact_variant"]
    assert REQUIRED_BINARY_METRICS <= set(transfer["evaluation"]["secondary_metrics"])
    assert REQUIRED_ROW_LEVEL_FIELDS <= set(transfer["evaluation"]["retain_row_level"])
    assert "fit_materials" not in alignment


def test_panel_composes_balanced_fit_and_same_task_reference(study) -> None:
    config = materialize_stage(select_task(study, "ag_news", "pooled_equal"), "align")
    fit = config["fit_materials"]

    assert fit["expected_train_rows"] == 12000
    assert fit["task_balanced"] is True
    assert fit["evaluation_included"] is False
    assert [entry["fit_rows"] for entry in fit["datasets"]] == [6000, 6000]
    assert [entry["dataset_key"] for entry in fit["datasets"]] == [
        "sst2-sentiment-v1",
        "wildguardmix-v1",
    ]
    assert config["reference_materials"]["source_variant"] == "ag-news-align"
    assert stage_prefix(config).endswith("/pooled-equal-fit-ag-news-eval")


def test_panel_included_task_compatibility_is_explicit(study) -> None:
    config = materialize_stage(select_task(study, "sst2", "pooled_full"), "align")

    assert config["fit_materials"]["evaluation_included"] is True
    assert config["fit_materials"]["expected_train_rows"] == 24000


def test_panel_variants_have_unique_results_and_shared_activations(study) -> None:
    variants = list(task_variants(study, "align"))
    configs = [materialize_stage(variant, "align") for variant in variants]

    assert len(configs) == 18
    assert len({stage_prefix(config) for config in configs}) == len(configs)
    target = [config for config in configs if config["task"] == "ag_news"]
    assert len({activation_prefix(config, "smollm1") for config in target}) == 1
    assert len(list(task_variants(study, "extract"))) == 4


def test_panel_rejects_heldout_fit_data(study) -> None:
    study["fit_conditions"]["leaked"] = {"sst2": 6000, "mnli": 6000}

    with pytest.raises(ConfigError, match="Held-out task activations"):
        select_task(study, "ag_news", "leaked")


@pytest.mark.parametrize("rows", [True, 1, 12001, 12.5])
def test_panel_rejects_invalid_fit_budgets(study, rows) -> None:
    study["fit_conditions"]["bad"] = {"sst2": rows}

    with pytest.raises(ConfigError, match="Panel fit rows"):
        select_task(study, "ag_news", "bad")


def test_panel_rejects_task_specific_model_changes(study) -> None:
    study["tasks"]["ag_news"]["models"] = {}

    with pytest.raises(ConfigError, match="Task overrides"):
        select_task(study, "ag_news")


def test_panel_requires_pinned_task_data(study) -> None:
    study["tasks"]["ag_news"]["dataset"]["revision"] = "main"

    with pytest.raises(ConfigError, match="40-character"):
        select_task(study, "ag_news")


def test_original_single_task_studies_remain_unchanged() -> None:
    study = load_config(CONFIGS_DIR / "studies" / "olmo1_independent_training_mnli.yaml")

    assert select_task(study) is study
    assert list(task_variants(study, "align")) == [study]
    with pytest.raises(ConfigError, match="configured task panel"):
        select_task(study, "mnli")
