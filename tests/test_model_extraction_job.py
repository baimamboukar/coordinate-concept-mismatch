import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from torch import nn

from probe_transfer.extraction.job import run_extraction_job
from probe_transfer.extraction.runner import run_model_extraction


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def __call__(
        self,
        prompts,
        *,
        padding=False,
        truncation=False,
        max_length=None,
        return_tensors=None,
        return_length=False,
    ):
        token_ids = [list(range(1, len(prompt.split()) + 1)) for prompt in prompts]
        lengths = [len(tokens) for tokens in token_ids]
        if truncation:
            token_ids = [tokens[:max_length] for tokens in token_ids]
        if return_tensors == "pt":
            width = max(len(tokens) for tokens in token_ids)
            padded = [tokens + [0] * (width - len(tokens)) for tokens in token_ids]
            masks = [[1] * len(tokens) + [0] * (width - len(tokens)) for tokens in token_ids]
            return {"input_ids": torch.tensor(padded), "attention_mask": torch.tensor(masks)}
        output: dict[str, list[list[int]] | list[int]] = {"input_ids": token_ids}
        if return_length:
            output["length"] = lengths
        return output


class FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(32, 3)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids, attention_mask, **_):
        base = input_ids.float().unsqueeze(-1).repeat(1, 1, 3)
        return SimpleNamespace(hidden_states=tuple(base + index for index in range(5)))


class Tracker:
    def __init__(self) -> None:
        self.logged = {}
        self.reports = []

    def metrics(self, values) -> None:
        self.logged.update(values)

    def report(self, title, body) -> None:
        self.reports.append((title, body))


def _config(rows_dir: Path, output_dir: Path) -> dict:
    return {
        "stage": "extract",
        "data_seeds": [42, 137],
        "models": {
            "llama": {
                "id": "fake/llama",
                "revision": "a" * 40,
                "layers": 4,
                "hidden_size": 3,
            }
        },
        "dataset": {"id": "fake/data", "revision": "b" * 40},
        "sampling": {"train_size": 2, "validation_size": 2, "test_size": 2},
        "activations": {
            "prompt_format": "raw",
            "token_position": "last_non_padding",
            "normalized_depths": [0.25, 0.5, 0.75, 1.0],
            "max_length": 3,
            "max_truncation_rate": 0.5,
            "batch_size": 2,
            "dtype": "bfloat16",
            "storage_dtype": "bfloat16",
        },
        "extraction": {
            "mode": "full",
            "models": ["llama"],
            "rows_dir": str(rows_dir),
            "staging_dir": str(output_dir),
            "repeatability_rows": 1,
            "repeatability_atol": 0.0,
        },
    }


def _write_rows(path: Path, offset: int) -> None:
    rows = [
        {"row_id": offset, "prompt": "one two", "label": 0, "adversarial": False},
        {
            "row_id": offset + 1,
            "prompt": "one two three four",
            "label": 1,
            "adversarial": True,
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _prepared_data(root: Path) -> None:
    _write_rows(root / "test.jsonl", 0)
    for seed in (42, 137):
        _write_rows(root / f"seed_{seed}_train.jsonl", seed * 10)
        _write_rows(root / f"seed_{seed}_validation.jsonl", seed * 10 + 2)


def test_extracts_one_model_into_mergeable_flat_layout(tmp_path: Path) -> None:
    rows_dir = tmp_path / "rows"
    output_dir = tmp_path / "staged"
    _prepared_data(rows_dir)

    completion = run_extraction_job(
        _config(rows_dir, output_dir),
        model_name="llama",
        model_loader=lambda *_args, **_kwargs: (FakeTokenizer(), FakeModel()),
    )

    model_dir = output_dir / "activations" / "llama"
    names = {
        "test.safetensors",
        "seed_42_train.safetensors",
        "seed_42_validation.safetensors",
        "seed_137_train.safetensors",
        "seed_137_validation.safetensors",
    }
    assert completion.status == "complete"
    assert len(completion.splits) == 5
    assert all((model_dir / name).is_file() for name in names)
    assert {path.name for path in model_dir.iterdir()} == names
    with safe_open(model_dir / "seed_42_train.safetensors", framework="pt") as saved:
        assert saved.get_tensor("row_ids").tolist() == [420, 421]
        assert saved.get_tensor("labels").tolist() == [0, 1]
        assert saved.get_slice("layer_75").get_shape() == [2, 3]


def test_enforces_test_size_and_truncation_before_completion(tmp_path: Path) -> None:
    rows_dir = tmp_path / "rows"
    output_dir = tmp_path / "staged"
    _prepared_data(rows_dir)
    config = _config(rows_dir, output_dir)
    config["sampling"]["test_size"] = 4
    loaded = False

    def loader(*_args, **_kwargs):
        nonlocal loaded
        loaded = True
        return FakeTokenizer(), FakeModel()

    with pytest.raises(ValueError, match="Expected 4 prepared rows"):
        run_extraction_job(
            config,
            model_name="llama",
            model_loader=loader,
        )
    assert loaded is False
    assert not (output_dir / "activations" / "llama").exists()


def test_failed_extraction_leaves_no_partial_model_directory(tmp_path: Path) -> None:
    rows_dir = tmp_path / "rows"
    output_dir = tmp_path / "staged"
    _prepared_data(rows_dir)

    class FailingModel(FakeModel):
        def forward(self, *_args, **_kwargs):
            raise RuntimeError("forward failed")

    with pytest.raises(RuntimeError, match="forward failed"):
        run_extraction_job(
            _config(rows_dir, output_dir),
            model_name="llama",
            model_loader=lambda *_args, **_kwargs: (FakeTokenizer(), FailingModel()),
        )

    activation_root = output_dir / "activations"
    assert not (activation_root / "llama").exists()
    assert not list(activation_root.glob(".llama-*"))


def test_runner_selects_one_enabled_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config(tmp_path / "rows", tmp_path / "output")
    completion = SimpleNamespace(splits=(SimpleNamespace(rows=10, truncated_rows=1),))
    selected = []

    def fake_job(_config, *, model_name):
        selected.append(model_name)
        return completion

    monkeypatch.setattr("probe_transfer.extraction.runner.run_extraction_job", fake_job)
    tracker = Tracker()

    assert run_model_extraction(config, tracker, "llama") is completion
    assert selected == ["llama"]
    assert tracker.logged == {"extraction/rows": 10.0, "extraction/truncation_rate": 0.1}


def test_runner_rejects_an_unconfigured_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path / "rows", tmp_path / "output")

    with pytest.raises(ValueError, match="not enabled"):
        run_model_extraction(config, Tracker(), "qwen")
