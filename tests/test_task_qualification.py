import json

import pytest

from core.config import load_config
from core.constants import CONFIGS_DIR
from pipeline.config import materialize_stage
from pipeline.panel import select_task
from pipeline.qualification import qualify_task


def _study():
    return load_config(
        CONFIGS_DIR / "studies/smollm_pairing_specific_independent_confirmation.yaml"
    )


def _write_rows(root, *, failures: int = 4, missing: bool = False) -> None:
    study = _study()
    config = materialize_stage(select_task(study, "qnli"), "align")
    alignment = config["alignment"]
    pairs = config["evaluation"]["pair_groups"]["primary"]
    contexts = [(seed, source, target) for seed in config["data_seeds"] for source, target in pairs]
    gaps, recoveries = [], []
    for index, (seed, source, target) in enumerate(contexts):
        common = {
            "data_seed": seed,
            "depth": alignment["primary_depth"],
            "probe_family": alignment["primary_probe_family"],
            "source_model": source,
            "target_model": target,
            "pair_group": "primary",
        }
        gaps.append({**common, "transfer_failed": index < failures})
        recoveries.extend(
            [
                {
                    **common,
                    "method": alignment["primary_method"],
                    "recovery_fraction": 0.8,
                    "substantial_recovery": True,
                },
                {
                    **common,
                    "method": alignment["negative_control"],
                    "recovery_fraction": 0.1,
                    "substantial_recovery": False,
                },
            ]
        )
    if missing:
        recoveries.pop()
    paths = (
        (root / "materials/qnli/results/transfer_gaps.jsonl", gaps),
        (root / "qnli/same_task/results/recovery.jsonl", recoveries),
    )
    for path, rows in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def test_fresh_task_qualification_applies_all_locked_gates(tmp_path) -> None:
    _write_rows(tmp_path)
    result = qualify_task(tmp_path, _study(), "qnli")

    assert result == {
        "frozen_failures": 4,
        "median_same_task_recovery": 0.8,
        "same_task_substantial": 4,
        "shuffled_substantial": 0,
        "qualified": True,
    }

    _write_rows(tmp_path, failures=3)
    assert qualify_task(tmp_path, _study(), "qnli")["qualified"] is False


def test_fresh_task_qualification_rejects_incomplete_contexts(tmp_path) -> None:
    _write_rows(tmp_path, missing=True)
    with pytest.raises(ValueError, match="incomplete"):
        qualify_task(tmp_path, _study(), "qnli")
