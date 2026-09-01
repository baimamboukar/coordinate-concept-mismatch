import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from core.config import ConfigError
from core.constants import (
    ACTIVATION_ROWS_ENV,
    ACTIVATION_STAGING_ENV,
    BASELINE_ARTIFACT_ENV,
    EXPERIMENT_OUTPUT_ENV,
    PROJECT_ROOT,
)
from pipeline.config import materialize_stage
from pipeline.panel import select_task
from probe_transfer.alignment.runner import _assert_expected_outputs
from probe_transfer.artifacts import write_json
from probe_transfer.materialization import verify_prior_probe_splits
from probe_transfer.transfer.runner import _validate_activations, _validate_outputs


def validate_material_preparation(study: dict[str, Any]) -> None:
    enabled = study.get("execution", {}).get("prepare_materials", False)
    if type(enabled) is not bool:
        raise ConfigError("prepare_materials must be a boolean.")
    if not enabled:
        return
    stages = study["pipeline"]["stages"]
    expected = {
        "study": study["name"],
        "transfer": stages["transfer"]["name"],
        "align": stages["align"]["name"],
    }
    if study["reuse_materials"] != expected:
        raise ConfigError("Generated panel materials must belong to the current study.")


def prepare_panel_materials(
    study: dict[str, Any], path: Path, root: Path, tasks: list[str]
) -> None:
    validate_material_preparation(study)
    for task in tasks:
        _run_stage(study, path, root, task, "prepare")
    audit = root / "results"
    audit.mkdir(exist_ok=True)
    for task in tasks:
        source = root / "rows" / task / "split_audit.json"
        config = materialize_stage(select_task(study, task), "prepare")
        verified = verify_prior_probe_splits(config, source.parent, root / "prior_splits" / task)
        if source.is_file():
            record = json.loads(source.read_text())
            record["partition"]["prior_split_files_verified"] = verified
            write_json(audit / f"{task}_split_audit.json", record)
    for stage in ("preflight", "extract"):
        for task in tasks:
            for model in study["extraction"]["models"]:
                _run_stage(study, path, root, task, stage, model)
    for task in tasks:
        for stage in ("transfer", "align"):
            _run_stage(study, path, root, task, stage)


def _run_stage(study, path, root, task, stage, model=None) -> None:
    materials = root / "materials" / task
    materials.mkdir(parents=True, exist_ok=True)
    rows = root / "rows" / task
    output = root / task / "same_task"
    output.mkdir(parents=True, exist_ok=True)
    environment = {
        **os.environ,
        ACTIVATION_ROWS_ENV: str(rows),
        ACTIVATION_STAGING_ENV: str(materials),
        BASELINE_ARTIFACT_ENV: str(materials),
        EXPERIMENT_OUTPUT_ENV: str(output),
    }
    command = [
        sys.executable,
        str(PROJECT_ROOT / "src/run.py"),
        str(path.resolve()),
        stage,
        "--task",
        task,
    ]
    if model is not None:
        command.extend(["--model", model])
    config = materialize_stage(select_task(study, task), stage)
    if stage == "extract" and (materials / "activations" / str(model)).is_dir():
        _validate_activations(materials, {**config, "models": {model: config["models"][model]}})
        command.append("--publish-only")
    elif stage == "transfer" and (materials / "results").is_dir():
        _validate_outputs(materials, config)
        command.append("--publish-only")
    elif stage == "align" and (output / "results").is_dir():
        _assert_expected_outputs(output, config)
        command.append("--publish-only")
    print(f"Preparing {study['name']}: {task}/{stage}" + (f"/{model}" if model else ""), flush=True)
    with (materials / "stages.log").open("a") as log:
        subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
