import pytest

from pipeline.confirmation import (
    endpoints_from_summary,
    holm_adjust,
    synthesize_independent_confirmation,
)

RULES = {
    "total_endpoints": 4,
    "minimum_endpoint_passes": 3,
    "require_each_task": True,
    "require_each_model_pair": True,
    "multiple_testing": "holm",
    "familywise_alpha": 0.05,
}


def _endpoint(pair: str, task: str, probability: float = 0.01, qualified: bool = True):
    return {
        "model_pair": pair,
        "task": task,
        "qualified": qualified,
        "passes_criterion": qualified,
        "pooled_empirical_p": probability if qualified else None,
    }


def test_confirmation_applies_holm_and_coverage() -> None:
    endpoints = [
        _endpoint("smollm", "qnli"),
        _endpoint("smollm", "qqp"),
        _endpoint("olmo", "qnli"),
        _endpoint("olmo", "qqp", qualified=False),
    ]
    result = synthesize_independent_confirmation(endpoints, RULES)

    assert result["confirmed"] is True
    assert result["endpoint_passes"] == 3
    assert all(row["holm_adjusted_p"] == pytest.approx(0.04) for row in result["endpoints"][:3])
    assert result["endpoints"][3]["pooled_empirical_p"] == 1.0


def test_confirmation_rejects_fewer_than_three_endpoints() -> None:
    endpoints = [
        _endpoint("smollm", "qnli"),
        _endpoint("smollm", "qqp", probability=0.2),
        _endpoint("olmo", "qnli"),
        _endpoint("olmo", "qqp", qualified=False),
    ]
    endpoints[1]["passes_criterion"] = False

    assert synthesize_independent_confirmation(endpoints, RULES)["confirmed"] is False


def test_summary_conversion_retains_a_skipped_endpoint() -> None:
    summary = {
        "qualifications": {"qnli": {"qualified": True}, "qqp": {"qualified": False}},
        "comparisons": [{"task": "qnli", "passes_criterion": True, "pooled_empirical_p": 0.01}],
    }
    endpoints = endpoints_from_summary("olmo", summary, ["qnli", "qqp"])

    assert endpoints[0]["pooled_empirical_p"] == 0.01
    assert endpoints[1]["qualified"] is False


def test_holm_adjustment_is_monotone_in_sorted_order() -> None:
    assert holm_adjust([0.01, 0.02, 0.03, 0.04]) == pytest.approx([0.04, 0.06, 0.06, 0.06])
