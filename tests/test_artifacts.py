import json
from pathlib import Path

import numpy as np
from safetensors import safe_open

from probe_transfer.artifacts import save_probe_bundle, write_jsonl
from probe_transfer.probes import train_linear_probe


def test_probe_bundle_is_safe_and_self_describing(tmp_path: Path) -> None:
    train_x = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    train_y = np.array([0, 0, 1, 1])
    probe = train_linear_probe(train_x, train_y, train_x, train_y, c_values=[1.0])
    path = tmp_path / "probes.safetensors"

    digest = save_probe_bundle(path, {"layer_75.linear": probe}, {"layer_75.linear": {}})

    with safe_open(path, framework="pt", device="cpu") as stored:
        details = json.loads(stored.metadata()["probes"])
        assert "layer_75.linear.coefficient" in set(stored.keys())
    assert details["layer_75.linear"]["kind"] == "linear"
    assert len(digest) == 64


def test_jsonl_writer_is_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"

    first = write_jsonl(path, [{"b": 2, "a": 1}])
    second = write_jsonl(path, [{"a": 1, "b": 2}])

    assert first == second
