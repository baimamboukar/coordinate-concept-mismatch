from core.config import load_config
from core.constants import CONFIGS_DIR
from pipeline.config import materialize_stage


def test_controlled_symmetry_protocol_is_derived() -> None:
    study = load_config(CONFIGS_DIR / "studies" / "pythia_controls.yaml")
    config = materialize_stage(study, "symmetry")

    assert config["symmetry"]["transformation"] == "residual_permutation"
    assert config["symmetry"]["models"] == ["pythia_seed1234", "pythia_seed1"]
    assert config["symmetry"]["transformation_seeds"] == [42, 137]
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


def test_mlp_neuron_symmetry_protocol_is_site_aware_and_derived() -> None:
    study = load_config(CONFIGS_DIR / "studies" / "modern_mlp_neuron_symmetry.yaml")
    transfer = materialize_stage(study, "transfer")
    symmetry = materialize_stage(study, "symmetry")

    assert transfer["activations"]["site"] == "mlp_intermediate"
    assert transfer["expected_outputs"] == {
        "metrics_rows": 6,
        "prediction_rows": 10194,
        "transfer_gap_rows": 0,
        "probe_bundles": 2,
    }
    assert symmetry["symmetry"]["transformation"] == "mlp_neuron_permutation"
    assert symmetry["symmetry"]["width"] == 14336
    assert symmetry["symmetry"]["estimated_alignment"]["method"] == "exact_permutation"
    assert symmetry["expected_outputs"] == {
        "metrics_rows": 60,
        "prediction_rows": 101940,
        "recovery_rows": 12,
        "function_gate_rows": 3,
        "probe_bundles": 4,
        "function_smoke_gate_rows": 3,
        "alignment_diagnostic_rows": 4,
    }


def test_attention_head_symmetry_protocol_is_site_aware_and_derived() -> None:
    study = load_config(CONFIGS_DIR / "studies" / "modern_attention_head_symmetry.yaml")
    transfer = materialize_stage(study, "transfer")
    symmetry = materialize_stage(study, "symmetry")

    assert transfer["activations"]["site"] == "attention_output"
    assert transfer["expected_outputs"] == {
        "metrics_rows": 6,
        "prediction_rows": 10194,
        "transfer_gap_rows": 0,
        "probe_bundles": 2,
    }
    assert symmetry["symmetry"]["transformation"] == "attention_head_permutation"
    assert symmetry["symmetry"]["attention_layout"] == {
        "query_heads": 32,
        "key_value_heads": 8,
        "head_dim": 128,
    }
    assert symmetry["expected_outputs"] == {
        "metrics_rows": 60,
        "prediction_rows": 101940,
        "recovery_rows": 12,
        "function_gate_rows": 3,
        "probe_bundles": 4,
        "function_smoke_gate_rows": 3,
        "alignment_diagnostic_rows": 4,
    }


def test_positive_diagonal_symmetry_reuses_mlp_baseline() -> None:
    study = load_config(CONFIGS_DIR / "studies" / "modern_mlp_positive_diagonal_symmetry.yaml")
    config = materialize_stage(study, "symmetry")

    assert config["activations"]["site"] == "mlp_intermediate"
    assert config["materials"]["source_study"] == "modern_mlp_neuron_symmetry"
    assert config["symmetry"]["transformation"] == "mlp_positive_diagonal"
    assert config["symmetry"]["scale_range"] == [0.125, 8.0]
    assert config["symmetry"]["estimated_alignment"]["method"] == "positive_diagonal"
    assert config["expected_outputs"] == {
        "metrics_rows": 60,
        "prediction_rows": 101940,
        "recovery_rows": 12,
        "function_gate_rows": 3,
        "probe_bundles": 4,
        "function_smoke_gate_rows": 3,
        "alignment_diagnostic_rows": 4,
    }
