from pathlib import Path

from core.config import load_config
from core.constants import CONFIGS_DIR, PIPELINE_STAGES
from pipeline.config import materialize_stage
from pipeline.runner import run_stage
from pipeline.stages import HANDLERS


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
