from pathlib import Path

import pytest

from core.config import ConfigError, load_config
from core.constants import CONFIGS_DIR, REQUIRED_BINARY_METRICS, REQUIRED_ROW_LEVEL_FIELDS
from pipeline.config import materialize_stage


def write_study(path: Path, name: str) -> None:
    path.write_text(
        f"name: {name}\nseed: 42\ndata_seeds: [42, 137]\npipeline:\n  stages:\n    prepare: {{}}\n"
    )


def test_loads_semantically_named_study(tmp_path: Path) -> None:
    path = tmp_path / "modern_models.yaml"
    write_study(path, path.stem)

    assert load_config(path)["name"] == "modern_models"


def test_rejects_number_only_study_name(tmp_path: Path) -> None:
    path = tmp_path / "exp_001.yaml"
    write_study(path, path.stem)

    with pytest.raises(ConfigError, match="Number-only"):
        load_config(path)


def test_requires_filename_to_match_name(tmp_path: Path) -> None:
    path = tmp_path / "modern_models.yaml"
    write_study(path, "different_study")

    with pytest.raises(ConfigError, match="filename"):
        load_config(path)


def test_composes_relative_defaults(tmp_path: Path) -> None:
    (tmp_path / "defaults.yaml").write_text("seed: 42\ndata_seeds: [42, 137]\n")
    path = tmp_path / "modern_models.yaml"
    path.write_text(
        "extends: defaults.yaml\nname: modern_models\npipeline:\n  stages:\n    prepare: {}\n"
    )

    config = load_config(path)

    assert config["seed"] == 42
    assert config["data_seeds"] == [42, 137]
    assert "extends" not in config


def test_huggingface_resources_require_commit_revision(tmp_path: Path) -> None:
    path = tmp_path / "modern_models.yaml"
    path.write_text(
        "name: modern_models\n"
        "seed: 42\n"
        "data_seeds: [42, 137]\n"
        "models:\n"
        "  source: {id: organization/model, revision: main}\n"
        "pipeline:\n"
        "  stages:\n"
        "    prepare: {}\n"
    )

    with pytest.raises(ConfigError, match="40-character commit"):
        load_config(path)


def test_modern_transfer_is_a_derived_stage_not_a_python_runner() -> None:
    study = load_config(CONFIGS_DIR / "studies" / "modern_models.yaml")
    config = materialize_stage(study, "transfer")

    assert "runner" not in study
    assert config["study"] == "modern_models"
    assert config["name"] == "frozen_probe_transfer_baseline"
    assert list(config["models"]) == ["llama", "qwen", "nemotron", "mistral", "granite"]
    assert config["expected_outputs"] == {
        "metrics_rows": 300,
        "prediction_rows": 509700,
        "transfer_gap_rows": 240,
        "probe_bundles": 10,
    }
    assert REQUIRED_BINARY_METRICS.issubset(config["evaluation"]["secondary_metrics"])
    assert REQUIRED_ROW_LEVEL_FIELDS.issubset(config["evaluation"]["retain_row_level"])


def test_pythia_uses_the_same_transfer_stage() -> None:
    study = load_config(CONFIGS_DIR / "studies" / "pythia_controls.yaml")
    config = materialize_stage(study, "transfer")

    assert config["name"] == "frozen_probe_transfer_baseline"
    assert config["stage"] == "transfer"
    assert config["expected_outputs"] == {
        "metrics_rows": 48,
        "prediction_rows": 81552,
        "transfer_gap_rows": 24,
        "probe_bundles": 4,
    }


def test_stage_requires_artifact_contract(tmp_path: Path) -> None:
    path = tmp_path / "missing_artifacts.yaml"
    write_study(path, path.stem)
    study = load_config(path)

    with pytest.raises(ConfigError, match="Hugging Face bucket"):
        materialize_stage(study, "prepare")
