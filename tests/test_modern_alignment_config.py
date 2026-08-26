from core.config import load_config
from core.constants import CONFIGS_DIR
from pipeline.config import materialize_stage
from probe_transfer.alignment.materials import direction_groups


def test_modern_alignment_protocol_is_composed_from_the_study() -> None:
    study = load_config(CONFIGS_DIR / "studies" / "modern_models.yaml")
    config = materialize_stage(study, "align")

    assert len(direction_groups(config)) == 16
    assert direction_groups(config)[:2] == [
        ("llama", "qwen", "primary"),
        ("qwen", "llama", "primary"),
    ]
    assert config["alignment"]["primary_probe_family"] == "linear"
    assert config["evaluation"]["bootstrap_samples"] == 2000
    assert config["expected_outputs"] == {
        "metrics_rows": 704,
        "prediction_rows": 1196096,
        "recovery_rows": 512,
        "alignment_diagnostic_rows": 192,
    }
    assert config["artifact_variant"] == "combined"


def test_two_model_alignment_retains_bidirectional_primary_default() -> None:
    config = {"models": {"first": {}, "second": {}}, "evaluation": {}}

    assert direction_groups(config) == [
        ("first", "second", "primary"),
        ("second", "first", "primary"),
    ]
