from pathlib import Path

import pytest

from core.config import load_config
from core.constants import CONFIGS_DIR, PIPELINE_STAGES
from pipeline.config import materialize_stage
from pipeline.runner import run_stage
from pipeline.stages import HANDLERS
from probe_transfer.symmetry.protocol import selected_models


def test_pipeline_has_one_stable_handler_per_stage() -> None:
    assert tuple(HANDLERS) == PIPELINE_STAGES


def test_data_preparation_is_shared_and_worker_local() -> None:
    study = load_config(CONFIGS_DIR / "studies" / "modern_models.yaml")
    config = materialize_stage(study, "prepare")

    assert config["name"] == "wildguardmix_data"
    assert config["tracking"] == {"wandb": False, "mode": "offline", "local_report": False}


def test_source_tree_has_no_experiment_specific_runners() -> None:
    root = Path(__file__).parents[1]
    experiment_dir = root / "src" / "experiments"

    assert not experiment_dir.exists() or not list(experiment_dir.glob("*.py"))


def test_publish_only_retries_without_running_compute(tmp_path: Path, monkeypatch) -> None:
    study = load_config(CONFIGS_DIR / "studies" / "pythia_controls.yaml")
    published = []
    monkeypatch.setenv("ACTIVATION_STAGING_DIR", str(tmp_path))
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    monkeypatch.setattr(
        "pipeline.runner.publish_artifacts",
        lambda _config, requests, _tracker: published.extend(requests),
    )
    monkeypatch.setitem(
        HANDLERS,
        "transfer",
        lambda *_args: (_ for _ in ()).throw(AssertionError("compute should not run")),
    )

    run_stage(study, "transfer", publish_only=True)

    assert [request.source.name for request in published] == ["probes", "results"]


def test_symmetry_worker_scopes_contract_and_artifact_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study = load_config(CONFIGS_DIR / "studies" / "modern_models.yaml")
    published = []
    monkeypatch.setenv("EXPERIMENT_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(
        "pipeline.runner.publish_artifacts",
        lambda _config, requests, _tracker: published.extend(requests),
    )

    config = run_stage(study, "symmetry", model="llama", publish_only=True)

    assert selected_models(config) == ["llama"]
    assert config["expected_outputs"]["metrics_rows"] == 60
    prefix = (
        "studies/modern-residual-permutation-probe-transport/modern-models/llama-3.1-8b-instruct"
    )
    assert [request.remote_prefix for request in published] == [
        f"{prefix}/probes",
        f"{prefix}/results",
    ]


def test_multi_model_symmetry_requires_worker_model() -> None:
    study = load_config(CONFIGS_DIR / "studies" / "modern_models.yaml")

    with pytest.raises(ValueError, match="require --model"):
        run_stage(study, "symmetry", publish_only=True)
