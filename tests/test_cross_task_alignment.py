import json
from pathlib import Path

import pytest

from probe_transfer.alignment.cross_task import (
    add_improvement_retention,
    load_recovery_reference,
)


def _row(**updates):
    row = {
        "data_seed": 42,
        "depth": 0.75,
        "probe_family": "linear",
        "source_model": "ai2",
        "target_model": "amd",
        "pair_group": "primary",
        "method": "affine_ridge",
        "raw_auroc_gap": 0.4,
        "aligned_auroc": 0.8,
        "aligned_auroc_improvement": 0.3,
        "recovery_fraction": 0.75,
    }
    return {**row, **updates}


def test_cross_task_recovery_records_same_task_retention(tmp_path: Path) -> None:
    path = tmp_path / "recovery.jsonl"
    path.write_text(json.dumps(_row()) + "\n")
    reference = load_recovery_reference(path)

    enriched = add_improvement_retention(
        _row(aligned_auroc=0.7, aligned_auroc_improvement=0.225, recovery_fraction=0.5625),
        reference,
        tolerance=1e-5,
    )

    assert enriched["same_task_aligned_auroc"] == 0.8
    assert enriched["same_task_recovery_fraction"] == 0.75
    assert enriched["improvement_retention"] == pytest.approx(0.75)


def test_cross_task_recovery_rejects_a_different_baseline(tmp_path: Path) -> None:
    path = tmp_path / "recovery.jsonl"
    path.write_text(json.dumps(_row()) + "\n")

    with pytest.raises(ValueError, match="baselines differ"):
        add_improvement_retention(
            _row(raw_auroc_gap=0.2),
            load_recovery_reference(path),
            tolerance=1e-5,
        )
