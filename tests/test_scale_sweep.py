from copy import deepcopy

import numpy as np
import pytest

from core.config import ConfigError, load_config
from core.constants import CONFIGS_DIR
from pipeline.config import materialize_stage
from pipeline.validation import validate_stage
from probe_transfer.symmetry.cases import build_transformation_cases
from probe_transfer.symmetry.coordinates import CoordinateTransform
from probe_transfer.symmetry.sweep import scale_sweep_metrics


def sweep_config() -> dict:
    study = load_config(CONFIGS_DIR / "studies" / "modern_mlp_positive_diagonal_scale_sweep.yaml")
    return materialize_stage(study, "symmetry")


def sentiment_config(stage: str = "symmetry") -> dict:
    study = load_config(CONFIGS_DIR / "studies" / "sentiment_positive_diagonal_scale_sweep.yaml")
    return materialize_stage(study, stage)


def test_scale_sweep_contract_is_derived() -> None:
    config = sweep_config()

    assert config["symmetry"]["dose_response"]["ordered_variants"] == [
        "mild",
        "moderate",
        "strong",
        "extreme",
    ]
    assert config["expected_outputs"] == {
        "metrics_rows": 204,
        "prediction_rows": 346596,
        "recovery_rows": 48,
        "function_gate_rows": 9,
        "probe_bundles": 16,
        "function_smoke_gate_rows": 9,
        "alignment_diagnostic_rows": 16,
    }


def test_sentiment_scale_sweep_reuses_the_protocol_with_new_task_materials() -> None:
    symmetry = sentiment_config()
    transfer = sentiment_config("transfer")

    assert symmetry["dataset"]["id"] == "stanfordnlp/sst2"
    assert symmetry["sampling"]["test_size"] == 872
    assert symmetry["symmetry"]["probe_sensitivity"]["target_family"] == "cp_degree_2"
    assert symmetry["expected_outputs"] == {
        "metrics_rows": 204,
        "prediction_rows": 177888,
        "recovery_rows": 48,
        "function_gate_rows": 9,
        "probe_bundles": 16,
        "function_smoke_gate_rows": 9,
        "alignment_diagnostic_rows": 16,
    }
    assert transfer["expected_outputs"] == {
        "metrics_rows": 6,
        "prediction_rows": 5232,
        "transfer_gap_rows": 0,
        "probe_bundles": 2,
    }


def test_scale_sweep_uses_paired_directions_and_stable_case_keys() -> None:
    cases = build_transformation_cases(sweep_config()["symmetry"])

    assert len(cases) == 8
    assert [case.key for case in cases] == [
        "scale_mild_42",
        "scale_mild_137",
        "scale_moderate_42",
        "scale_moderate_137",
        "scale_strong_42",
        "scale_strong_137",
        "scale_extreme_42",
        "scale_extreme_137",
    ]
    for seed in (42, 137):
        paired = [case for case in cases if case.seed == seed]
        directions = []
        for case in paired:
            assert case.scale_maximum is not None
            directions.append(np.log(case.coordinates.values.numpy()) / np.log(case.scale_maximum))
        for actual in directions[1:]:
            np.testing.assert_allclose(actual, directions[0], rtol=1e-14, atol=1e-14)


def test_relative_scale_cases_reach_each_absolute_basis() -> None:
    cases = build_transformation_cases(sweep_config()["symmetry"])
    source = np.random.default_rng(17).normal(size=(3, 14336))
    current = CoordinateTransform.identity("positive_diagonal", source.shape[1])
    actual = source

    for case in cases:
        actual = case.coordinates.relative_from(current).apply_array(actual)
        np.testing.assert_allclose(
            actual,
            case.coordinates.apply_array(source),
            rtol=1e-12,
            atol=1e-12,
        )
        current = case.coordinates


def test_scale_sweep_summary_applies_prespecified_rules() -> None:
    recoveries = []
    gaps = {"mild": 0.01, "moderate": 0.03, "strong": 0.12, "extreme": 0.20}
    for data_seed in (42, 137):
        for family in ("linear", "cp_degree_2", "mlp"):
            for transformation_seed in (42, 137):
                for variant, gap in gaps.items():
                    recoveries.append(
                        {
                            "data_seed": data_seed,
                            "model": "mistral",
                            "depth": 0.75,
                            "probe_family": family,
                            "transformation_seed": transformation_seed,
                            "transformation_variant": variant,
                            "raw_auroc_gap": gap,
                            "coordinate_failure": gap >= 0.10,
                            "exact_recovery": True,
                            "estimated_recovery": True,
                        }
                    )

    metrics = scale_sweep_metrics(recoveries, sweep_config()["symmetry"])

    assert metrics["scale_sweep/dose_response_supported"] == 1.0
    assert metrics["scale_sweep/monotonic_trajectory_fraction"] == 1.0
    assert metrics["scale_sweep/first_robust_failure_index"] == 2.0
    assert metrics["scale_sweep/strong/coordinate_failure_fraction"] == 1.0


def test_probe_family_sensitivity_applies_out_of_task_replication_rule() -> None:
    recoveries = []
    gaps = {
        "linear": [0.002, 0.02, 0.04, 0.06],
        "cp_degree_2": [0.003, 0.03, 0.15, 0.22],
        "mlp": [0.001, 0.01, 0.02, 0.03],
    }
    variants = ["mild", "moderate", "strong", "extreme"]
    for data_seed in (42, 137):
        for family, family_gaps in gaps.items():
            for transformation_seed in (42, 137):
                for variant, gap in zip(variants, family_gaps, strict=True):
                    recoveries.append(
                        {
                            "data_seed": data_seed,
                            "model": "mistral",
                            "depth": 0.75,
                            "probe_family": family,
                            "transformation_seed": transformation_seed,
                            "transformation_variant": variant,
                            "reference_auroc": 0.90,
                            "raw_auroc_gap": gap,
                            "coordinate_failure": family == "cp_degree_2" and gap >= 0.10,
                            "exact_recovery": True,
                            "estimated_recovery": True,
                        }
                    )

    metrics = scale_sweep_metrics(recoveries, sentiment_config()["symmetry"])

    assert metrics["scale_sweep/task_viability_supported"] == 1.0
    assert metrics["scale_sweep/probe_sensitivity_supported"] == 1.0
    assert metrics["scale_sweep/strong/target_family_failure_fraction"] == 1.0
    assert metrics["scale_sweep/strong/comparison_family_failure_fraction"] == 0.0
    assert metrics["scale_sweep/strong/target_gap_advantage"] == pytest.approx(0.11)


def test_scale_sweep_rejects_nonreciprocal_ranges() -> None:
    config = deepcopy(sweep_config())
    config["symmetry"]["scale_ranges"]["strong"] = [0.04, 32.0]

    with pytest.raises(ConfigError, match="reciprocal range"):
        validate_stage(config)


def test_probe_sensitivity_cannot_cherry_pick_comparison_families() -> None:
    config = deepcopy(sentiment_config())
    config["symmetry"]["probe_sensitivity"]["comparison_families"] = ["linear"]

    with pytest.raises(ConfigError, match="every other primary family"):
        validate_stage(config)
