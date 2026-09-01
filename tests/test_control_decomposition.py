import pytest

from probe_transfer.alignment.control_decomposition import summarize_control_decomposition
from probe_transfer.alignment.task_adaptation import (
    coral_method,
    residual_shuffle_method,
    source_shuffle_method,
)


def _config() -> dict:
    return {
        "data_seeds": [42, 137],
        "alignment": {
            "primary_method": "low_rank_r8_n256",
            "task_adaptation": {
                "confirmatory_rank": 8,
                "confirmatory_rows": 256,
                "controls": {
                    "residual_shuffle_repeats": 20,
                    "source_shuffle_repeats": 20,
                },
            },
        },
        "evaluation": {
            "primary_pair_group": "primary",
            "pair_groups": {"primary": [["a", "b"], ["b", "a"]]},
        },
    }


def _rules() -> dict:
    return {
        "minimum_median_recovery": 0.5,
        "minimum_median_retention": 0.75,
        "minimum_median_paired_advantage": 0.1,
        "maximum_empirical_p": 0.05,
        "minimum_control_wins": 3,
    }


def _rows() -> list[dict]:
    rows = []
    for seed in (42, 137):
        for source, target in (("a", "b"), ("b", "a")):
            context = {
                "data_seed": seed,
                "source_model": source,
                "target_model": target,
                "improvement_retention": 0.85,
                "aligned_auroc": 0.9,
                "substantial_recovery": True,
            }
            rows.append({**context, "method": "low_rank_r8_n256", "recovery_fraction": 0.8})
            rows.append({**context, "method": coral_method(8, 256), "recovery_fraction": 0.3})
            for repeat in range(20):
                rows.append(
                    {
                        **context,
                        "method": residual_shuffle_method(8, 256, repeat),
                        "recovery_fraction": 0.10 + repeat / 1000,
                    }
                )
                rows.append(
                    {
                        **context,
                        "method": source_shuffle_method(8, 256, repeat),
                        "recovery_fraction": 0.2,
                    }
                )
    return rows


def test_pairing_specific_summary_applies_the_locked_empirical_rule() -> None:
    result = summarize_control_decomposition(_rows(), _config(), _rules())

    assert result["median_recovery"] == 0.8
    assert result["pooled_empirical_p"] == pytest.approx(1 / 21)
    assert result["control_wins"] == 4
    assert len(result["control_decomposition"]) == 4
    assert result["passes_criterion"] is True


def test_pairing_specific_summary_rejects_a_strong_shuffle_null() -> None:
    rows = _rows()
    for row in rows:
        if row["method"].startswith("residual_shuffle"):
            row["recovery_fraction"] = 0.85

    result = summarize_control_decomposition(rows, _config(), _rules())

    assert result["pooled_empirical_p"] == 1.0
    assert result["control_wins"] == 0
    assert result["passes_criterion"] is False


def test_pairing_specific_summary_rejects_missing_repeats() -> None:
    rows = _rows()
    rows.pop()
    with pytest.raises(ValueError, match="Missing control-decomposition row"):
        summarize_control_decomposition(rows, _config(), _rules())
