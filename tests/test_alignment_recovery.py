import numpy as np

from probe_transfer.alignment.recovery import alignment_recovery_record


def test_alignment_recovery_reports_complete_recovery() -> None:
    labels = np.array([0] * 50 + [1] * 50)
    oracle = np.linspace(-2, 2, 100)
    raw = -oracle
    config = {
        "evaluation": {
            "bootstrap_samples": 50,
            "confidence_level": 0.95,
            "oracle_gate": 0.75,
            "minimum_raw_gap": 0.10,
            "minimum_improvement": 0.05,
            "minimum_recovery": 0.50,
        }
    }

    record = alignment_recovery_record(
        {"method": "known"},
        labels,
        oracle,
        raw,
        oracle,
        1.0,
        config,
        42,
    )

    assert record["recovery_fraction"] == 1.0
    assert record["residual_auroc_gap"] == 0.0
    assert record["improvement_ci_lower"] > 0
    assert record["substantial_recovery"] is True


def test_repeated_controls_use_the_bounded_bootstrap_budget(monkeypatch) -> None:
    captured = []

    def intervals(*_args, **kwargs):
        captured.append(kwargs["samples"])
        return {
            "improvement_ci_lower": 0.1,
            "improvement_ci_upper": 0.2,
            "residual_ci_lower": 0.0,
            "residual_ci_upper": 0.1,
            "recovery_ci_lower": 0.5,
            "recovery_ci_upper": 0.9,
        }

    monkeypatch.setattr("probe_transfer.alignment.recovery._bootstrap_intervals", intervals)
    labels = np.array([0, 0, 1, 1])
    oracle = np.array([0.1, 0.2, 0.8, 0.9])
    raw = np.array([0.8, 0.7, 0.2, 0.1])
    config = {
        "evaluation": {
            "bootstrap_samples": 2000,
            "control_bootstrap_samples": 200,
            "confidence_level": 0.95,
            "oracle_gate": 0.75,
            "minimum_raw_gap": 0.10,
            "minimum_improvement": 0.05,
            "minimum_recovery": 0.50,
        }
    }
    alignment_recovery_record(
        {"method": "residual_shuffle_low_rank_r8_n256_rep00"},
        labels,
        oracle,
        raw,
        oracle,
        1.0,
        config,
        42,
    )
    assert captured == [200]
