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


def test_olmo2_seed_decomposition_is_task_replicated_and_counted() -> None:
    expected = {
        "olmo2_seed_decomposition_wildguard": {
            "transfer": {
                "metrics_rows": 192,
                "prediction_rows": 326208,
                "transfer_gap_rows": 144,
                "probe_bundles": 8,
            },
            "align": {
                "metrics_rows": 552,
                "prediction_rows": 937848,
                "recovery_rows": 408,
                "alignment_diagnostic_rows": 288,
            },
        },
        "olmo2_seed_decomposition_sst2": {
            "transfer": {
                "metrics_rows": 192,
                "prediction_rows": 167424,
                "transfer_gap_rows": 144,
                "probe_bundles": 8,
            },
            "align": {
                "metrics_rows": 552,
                "prediction_rows": 481344,
                "recovery_rows": 408,
                "alignment_diagnostic_rows": 288,
            },
        },
    }

    for study_name, counts in expected.items():
        study = load_config(CONFIGS_DIR / "studies" / f"{study_name}.yaml")
        transfer = materialize_stage(study, "transfer")
        alignment = materialize_stage(study, "align")

        assert len(study["models"]) == 4
        assert len(transfer["evaluation"]["pair_groups"]["primary"]) == 6
        assert len(transfer["evaluation"]["pair_groups"]["lineage_control"]) == 6
        assert len(direction_groups(alignment)) == 6
        assert alignment["alignment"]["depths"] == [0.25, 0.5, 0.75, 1.0]
        assert transfer["expected_outputs"] == counts["transfer"]
        assert alignment["expected_outputs"] == counts["align"]


def test_olmo2_stage_transition_alignment_reuses_the_fixed_baseline() -> None:
    study = load_config(CONFIGS_DIR / "studies" / "olmo2_stage_transition_alignment_sst2.yaml")
    alignment = materialize_stage(study, "align")
    directions = direction_groups(alignment)

    assert len(directions) == 6
    assert {group for _, _, group in directions} == {"lineage_control"}
    assert alignment["evaluation"]["primary_pair_group"] == "lineage_control"
    assert alignment["alignment"]["primary_depth"] == 0.75
    assert alignment["alignment"]["primary_probe_family"] == "linear"
    assert alignment["alignment"]["primary_method"] == "permutation_diagonal"
    assert alignment["materials"]["source_study"] == "olmo2_seed_decomposition_sst2"
    assert alignment["materials"]["source_variant"] == "sst2-baseline"
    assert alignment["artifact_variant"] == "sst2-lineage-alignment"
    assert alignment["expected_outputs"] == {
        "metrics_rows": 552,
        "prediction_rows": 481344,
        "recovery_rows": 408,
        "alignment_diagnostic_rows": 288,
    }


def test_olmo1_independent_training_is_task_replicated_with_shared_tokenizer() -> None:
    expected_predictions = {
        "olmo1_independent_training_sst2": (41856, 160448),
        "olmo1_independent_training_wildguard": (81552, 312616),
    }

    for study_name, (transfer_predictions, alignment_predictions) in expected_predictions.items():
        study = load_config(CONFIGS_DIR / "studies" / f"{study_name}.yaml")
        transfer = materialize_stage(study, "transfer")
        alignment = materialize_stage(study, "align")

        assert study["activations"]["add_special_tokens"] is False
        assert study["tokenizer"] == {
            "backend": "huggingface",
            "id": "allenai/OLMo-1B-hf",
            "revision": "aee7752d9c08ee4775e9b0091426d8410e8f6a89",
        }
        assert direction_groups(alignment) == [
            ("ai2_olmo1", "amd_olmo1", "primary"),
            ("amd_olmo1", "ai2_olmo1", "primary"),
        ]
        assert transfer["expected_outputs"] == {
            "metrics_rows": 48,
            "prediction_rows": transfer_predictions,
            "transfer_gap_rows": 24,
            "probe_bundles": 4,
        }
        assert alignment["expected_outputs"] == {
            "metrics_rows": 184,
            "prediction_rows": alignment_predictions,
            "recovery_rows": 136,
            "alignment_diagnostic_rows": 96,
        }


def test_olmo1_alignment_maps_are_transportable_across_tasks() -> None:
    expected = {
        "olmo1_map_transport_sst2_to_wildguard": (
            "sst2-sentiment-v1",
            "olmo1_independent_training_sst2",
            "sst2-fit-wildguard-eval",
            312616,
        ),
        "olmo1_map_transport_wildguard_to_sst2": (
            "wildguardmix-v1",
            "olmo1_independent_training_wildguard",
            "wildguard-fit-sst2-eval",
            160448,
        ),
    }

    for study_name, (dataset_key, source_study, variant, predictions) in expected.items():
        study = load_config(CONFIGS_DIR / "studies" / f"{study_name}.yaml")
        alignment = materialize_stage(study, "align")

        assert alignment["alignment"]["primary_method"] == "affine_ridge"
        assert alignment["fit_materials"] == {
            "dataset_key": dataset_key,
            "expected_train_rows": 12000,
            "source_study": source_study,
        }
        assert alignment["artifact_variant"] == variant
        assert alignment["reference_materials"]["source_name"] == "olmo1_independent_alignment"
        assert alignment["reference_materials"]["source_study"] != source_study
        assert alignment["expected_outputs"] == {
            "metrics_rows": 184,
            "prediction_rows": predictions,
            "recovery_rows": 136,
            "alignment_diagnostic_rows": 96,
        }
