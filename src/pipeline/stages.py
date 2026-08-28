import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from core.constants import BASELINE_ARTIFACT_ENV, EXPERIMENT_OUTPUT_ENV
from core.tracking import Tracker
from probe_transfer.alignment.runner import run_alignment_experiment
from probe_transfer.extraction.job import ROWS_ENV, STAGING_ENV
from probe_transfer.extraction.preflight import run_extraction_preflight
from probe_transfer.extraction.runner import run_model_extraction
from probe_transfer.layout import activation_prefix, model_artifact_key, stage_prefix
from probe_transfer.materialization import (
    materialize_activations,
    materialize_baseline,
    materialize_fit_activations,
    materialize_recovery_reference,
)
from probe_transfer.preparation import prepare_dataset
from probe_transfer.publication import Publication
from probe_transfer.symmetry.protocol import selected_models
from probe_transfer.symmetry.runner import run_symmetry_experiment
from probe_transfer.transfer.runner import run_staged_transfer

Handler = Callable[[dict[str, Any], Tracker, str | None], None]


def _prepare(config: dict[str, Any], tracker: Tracker, model: str | None) -> None:
    _reject_model(model)
    prepare_dataset(config, tracker)


def _preflight(config: dict[str, Any], tracker: Tracker, model: str | None) -> None:
    selected = _require_model(model)
    _ensure_prepared(config, tracker)
    run_extraction_preflight(config, tracker, selected)


def _extract(config: dict[str, Any], tracker: Tracker, model: str | None) -> None:
    selected = _require_model(model)
    _ensure_prepared(config, tracker)
    run_model_extraction(config, tracker, selected)


def _transfer(config: dict[str, Any], tracker: Tracker, model: str | None) -> None:
    _reject_model(model)
    output = _configured_path(config, STAGING_ENV, "staging_dir")
    materialize_activations(config, output)
    run_staged_transfer(config, tracker)


def _align(config: dict[str, Any], tracker: Tracker, model: str | None) -> None:
    _reject_model(model)
    baseline = _environment_path(BASELINE_ARTIFACT_ENV)
    materialize_baseline(config, baseline)
    fit_root = None
    reference_path = None
    if config.get("fit_materials") is not None:
        fit_root = baseline / "fit"
        materialize_fit_activations(config, fit_root)
        reference_path = materialize_recovery_reference(config, baseline / "reference")
    run_alignment_experiment(config, tracker, fit_root=fit_root, reference_path=reference_path)


def _symmetry(config: dict[str, Any], tracker: Tracker, model: str | None) -> None:
    _ensure_prepared(config, tracker)
    materialize_baseline(
        config,
        _environment_path(BASELINE_ARTIFACT_ENV),
        models=selected_models(config),
    )
    run_symmetry_experiment(config, tracker)


HANDLERS: dict[str, Handler] = {
    "prepare": _prepare,
    "preflight": _preflight,
    "extract": _extract,
    "transfer": _transfer,
    "align": _align,
    "symmetry": _symmetry,
}


def validate_invocation(config: dict[str, Any], stage: str, model: str | None) -> None:
    if stage in {"preflight", "extract"}:
        selected = _require_model(model)
        if selected not in config["extraction"]["models"]:
            raise ValueError(f"Model {selected} is not enabled for extraction.")
    elif stage == "symmetry":
        configured = selected_models(config)
        if model is None and len(configured) > 1:
            raise ValueError("Multi-model symmetry stages require --model.")
        if model is not None and model not in configured:
            raise ValueError(f"Model {model} is not enabled for symmetry.")
    else:
        _reject_model(model)


def publication_requests(
    config: dict[str, Any], stage: str, model: str | None
) -> list[Publication]:
    if stage in {"prepare", "preflight"}:
        return []
    if stage == "extract":
        selected = _require_model(model)
        root = _configured_path(config, STAGING_ENV, "staging_dir")
        return [Publication(root / "activations" / selected, activation_prefix(config, selected))]
    if stage == "transfer":
        root = _configured_path(config, STAGING_ENV, "staging_dir")
        prefix = stage_prefix(config)
        return [
            Publication(root / "probes", f"{prefix}/probes"),
            Publication(root / "results", f"{prefix}/results"),
        ]
    root = _environment_path(EXPERIMENT_OUTPUT_ENV)
    prefix = stage_prefix(config)
    if stage == "symmetry" and model is not None:
        prefix = f"{prefix}/{model_artifact_key(config, model)}"
    requests = [Publication(root / "results", f"{prefix}/results")]
    if stage == "symmetry":
        requests.insert(0, Publication(root / "probes", f"{prefix}/probes"))
    return requests


def _require_model(model: str | None) -> str:
    if not model:
        raise ValueError("This stage requires --model.")
    return model


def _reject_model(model: str | None) -> None:
    if model is not None:
        raise ValueError("--model is not valid for this stage.")


def _configured_path(config: dict[str, Any], environment: str, key: str) -> Path:
    value = os.getenv(environment) or config.get("extraction", {}).get(key)
    if not value:
        raise RuntimeError(f"{environment} is required.")
    return Path(value).expanduser().resolve()


def _environment_path(name: str) -> Path:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} is required.")
    return Path(value).expanduser().resolve()


def _ensure_prepared(config: dict[str, Any], tracker: Tracker) -> Path:
    root = _configured_path(config, ROWS_ENV, "rows_dir")
    expected = [root / "test.jsonl"]
    for seed in config["data_seeds"]:
        expected.extend([root / f"seed_{seed}_train.jsonl", root / f"seed_{seed}_validation.jsonl"])
    if all(path.is_file() for path in expected):
        return root
    if root.exists():
        raise FileExistsError(f"Prepared data directory is incomplete: {root}")
    return prepare_dataset(config, tracker)
