import numpy as np

from probe_transfer.alignment_recovery import alignment_recovery_record


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
