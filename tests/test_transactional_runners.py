from pathlib import Path

import pytest

from core.tracking import Tracker
from probe_transfer.alignment_runner import run_alignment_experiment
from probe_transfer.atomic import publish_directories
from probe_transfer.baseline_transfer import run_staged_transfer


def test_baseline_validation_failure_leaves_retryable_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "activations").mkdir()
    monkeypatch.setenv("ACTIVATION_STAGING_DIR", str(tmp_path))
    monkeypatch.setattr("probe_transfer.baseline_transfer._validate_activations", lambda *_: None)

    def fake_transfer(root, *_args):
        (root / "probes").mkdir()
        (root / "results").mkdir()
        return [], {}

    monkeypatch.setattr("probe_transfer.baseline_transfer.run_transfer", fake_transfer)
    monkeypatch.setattr(
        "probe_transfer.baseline_transfer._validate_outputs",
        lambda *_: (_ for _ in ()).throw(ValueError("late validation failed")),
    )

    with pytest.raises(ValueError, match="late validation failed"):
        run_staged_transfer({}, Tracker("test", tmp_path / "baseline-report.md"))

    assert not (tmp_path / "probes").exists()
    assert not (tmp_path / "results").exists()
    assert not list(tmp_path.glob(".transfer-*"))


def test_alignment_validation_failure_leaves_retryable_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = tmp_path / "baseline"
    output = tmp_path / "output"
    baseline.mkdir()
    output.mkdir()
    monkeypatch.setenv("BASELINE_ARTIFACT_DIR", str(baseline))
    monkeypatch.setenv("EXPERIMENT_OUTPUT_DIR", str(output))

    def fake_evaluation(_baseline, root, _config):
        (root / "results").mkdir()
        return [], [], {}

    monkeypatch.setattr(
        "probe_transfer.alignment_runner.evaluate_checkpoint_alignment", fake_evaluation
    )
    monkeypatch.setattr(
        "probe_transfer.alignment_runner._assert_expected_outputs",
        lambda *_: (_ for _ in ()).throw(ValueError("late validation failed")),
    )

    with pytest.raises(ValueError, match="late validation failed"):
        run_alignment_experiment({}, Tracker("test", tmp_path / "alignment-report.md"))

    assert not (output / "results").exists()
    assert not list(output.glob(".alignment-*"))


def test_directory_publication_rolls_back_partial_move(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging = tmp_path / "staging"
    destination = tmp_path / "destination"
    for root in (staging, destination):
        root.mkdir()
    for name in ("probes", "results"):
        (staging / name).mkdir()
    original_replace = Path.replace

    def fail_results(path: Path, target: Path):
        if path.name == "results":
            raise OSError("move failed")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_results)
    with pytest.raises(OSError, match="move failed"):
        publish_directories(staging, destination, ("probes", "results"))

    assert not (destination / "probes").exists()
    assert not (destination / "results").exists()
