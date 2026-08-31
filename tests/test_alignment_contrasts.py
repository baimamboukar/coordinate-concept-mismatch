import json

import pytest

from core.config import ConfigError
from probe_transfer.alignment.contrasts import condition_contrasts, validate_contrasts


def _study():
    return {
        "seed": 42,
        "evaluation": {"bootstrap_samples": 20, "confidence_level": 0.95},
        "fit_conditions": {"overlap": {}, "disjoint": {}},
        "decision_rules": {
            "contrasts": {
                "split": {
                    "reference": "overlap",
                    "treatment": "disjoint",
                    "primary": True,
                }
            }
        },
    }


def _write_results(root, name, scores):
    context = {
        "data_seed": 42,
        "depth": 0.75,
        "probe_family": "linear",
        "source_model": "s",
        "target_model": "t",
        "pair_group": "primary",
    }
    directory = root / "task" / name / "results"
    directory.mkdir(parents=True)
    rows = [
        {**context, "condition": "affine_ridge", "row_id": i, "label": i % 2, "score": score}
        for i, score in enumerate(scores)
    ]
    (directory / "predictions.jsonl").write_text(
        "\n".join(json.dumps(row) for row in reversed(rows))
    )
    recovery = {**context, "method": "affine_ridge", "raw_auroc_gap": 0.5, "recovery_fraction": 1.0}
    (directory / "recovery.jsonl").write_text(json.dumps(recovery) + "\n")


def test_paired_contrast_preserves_sign_and_row_identity(tmp_path) -> None:
    _write_results(tmp_path, "overlap", [1, 0, 1, 0])
    _write_results(tmp_path, "disjoint", [0, 1, 0, 1])
    (row,) = condition_contrasts(tmp_path, _study(), ["task"])
    assert row["auroc_change"] == row["ci_lower"] == row["ci_upper"] == 1.0
    path = tmp_path / "task/disjoint/results/predictions.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["label"] = 1 - rows[0]["label"]
    path.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError, match="identical row IDs and labels"):
        condition_contrasts(tmp_path, _study(), ["task"])


def test_contrast_requires_one_primary_and_enabled_distinct_conditions() -> None:
    study = _study()
    study["decision_rules"]["contrasts"]["split"]["reference"] = "missing"
    with pytest.raises(ConfigError, match="enabled fitting conditions"):
        validate_contrasts(study)
    study = _study()
    study["decision_rules"]["contrasts"]["split"]["primary"] = False
    with pytest.raises(ConfigError, match="exactly one primary"):
        validate_contrasts(study)
