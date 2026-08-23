from pathlib import Path

import pytest

from core.tracking import Tracker


def test_tracker_writes_local_log_and_report(tmp_path: Path) -> None:
    tracker = Tracker.start(
        {
            "name": "frozen_probe_transfer_baseline",
            "seed": 42,
            "tracking": {"wandb": False},
        },
        root=tmp_path,
    )

    tracker.metrics({"auroc": 0.75})
    tracker.report("Result", "Baseline completed.")
    tracker.finish()

    assert tracker.report_path.parent == tmp_path / "logs" / "frozen_probe_transfer_baseline"
    assert "Baseline completed." in tracker.report_path.read_text()


def test_training_requires_wandb(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must enable W&B"):
        Tracker.start(
            {
                "name": "frozen_probe_transfer_baseline",
                "seed": 42,
                "training": True,
                "tracking": {"wandb": False},
            },
            root=tmp_path,
        )
