import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.tracking import Tracker


def test_tracker_passes_declared_offline_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = {}

    def initialize(**parameters):
        captured.update(parameters)
        return SimpleNamespace()

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=initialize))
    Tracker.start(
        {
            "name": "frozen_probe_transfer_baseline",
            "stage": "modern_baseline",
            "seed": 42,
            "training": True,
            "tracking": {"wandb": True, "mode": "offline"},
        },
        root=tmp_path,
    )

    assert captured["mode"] == "offline"
