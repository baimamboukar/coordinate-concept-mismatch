from core.config import load_config
from core.constants import CONFIGS_DIR
from pipeline.config import materialize_stage


def test_controlled_symmetry_protocol_is_derived() -> None:
    study = load_config(CONFIGS_DIR / "studies" / "pythia_controls.yaml")
    config = materialize_stage(study, "symmetry")

    assert config["symmetry"]["transformation"] == "residual_permutation"
    assert config["symmetry"]["models"] == ["pythia_seed1234", "pythia_seed1"]
    assert config["symmetry"]["permutation_seeds"] == [42, 137]
    assert config["symmetry"]["gate_rows"] == 1699
    assert config["symmetry"]["gate_dtype"] == "float64"
    assert config["evaluation"]["primary_metrics"] == ["raw_auroc_gap", "recovery_fraction"]
    assert config["expected_outputs"] == {
        "metrics_rows": 192,
        "prediction_rows": 326208,
        "recovery_rows": 48,
        "function_gate_rows": 6,
        "probe_bundles": 8,
    }


def test_modern_symmetry_protocol_is_model_scoped_and_derived() -> None:
    study = load_config(CONFIGS_DIR / "studies" / "modern_models.yaml")
    config = materialize_stage(study, "symmetry")

    assert config["symmetry"]["models"] == ["mistral", "llama", "qwen"]
    assert config["execution"]["accelerators"] == ["H200"]
    assert config["symmetry"]["estimated_alignment"]["fit_rows"] == 2000
    assert config["expected_outputs"] == {
        "metrics_rows": 180,
        "prediction_rows": 305820,
        "recovery_rows": 36,
        "function_gate_rows": 9,
        "probe_bundles": 12,
        "function_smoke_gate_rows": 9,
        "alignment_diagnostic_rows": 12,
    }
