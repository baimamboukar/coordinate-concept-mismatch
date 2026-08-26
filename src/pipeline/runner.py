from pathlib import Path
from typing import Any

from core.constants import PROJECT_ROOT
from core.reproducibility import require_process_hash_seed, seed_everything
from core.tracking import Tracker
from pipeline.config import materialize_stage
from pipeline.stages import HANDLERS, publication_requests, validate_invocation
from probe_transfer.publication import publish_artifacts


def run_stage(
    study: dict[str, Any],
    stage: str,
    *,
    model: str | None = None,
    publish_only: bool = False,
    root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    config = materialize_stage(study, stage)
    validate_invocation(config, stage, model)
    if model is not None:
        config["worker_model"] = model
    requests = publication_requests(config, stage, model)
    if publish_only:
        if not requests:
            raise ValueError(f"Stage '{stage}' does not produce publishable artifacts.")
        publish_artifacts(config, requests, None)
        return config
    if config.get("deterministic", True):
        require_process_hash_seed(config["seed"])
    seed_everything(config["seed"], config.get("deterministic", True))
    tracker = Tracker.start(config, root=root)
    try:
        HANDLERS[stage](config, tracker, model)
        publish_artifacts(config, requests, tracker)
    except Exception as error:
        tracker.report("Failure", f"`{type(error).__name__}`: {error}")
        tracker.finish("failed")
        raise
    tracker.finish()
    return config
