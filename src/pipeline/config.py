from typing import Any

from core.config import ConfigError, merge_config
from pipeline.contracts import expected_outputs
from pipeline.validation import validate_stage


def materialize_stage(study: dict[str, Any], stage: str) -> dict[str, Any]:
    if "tasks" in study:
        raise ConfigError("Select a panel task before materializing a pipeline stage.")
    stages = study["pipeline"]["stages"]
    if stage not in stages:
        raise ConfigError(f"Stage '{stage}' is not configured for study '{study['name']}'.")

    base = {key: value for key, value in study.items() if key != "pipeline"}
    configured = merge_config(base, stages[stage])
    configured["study"] = study["name"]
    configured["name"] = stages[stage].get("name", f"{study['name']}_{stage}")
    configured["stage"] = stage
    configured.setdefault("training", False)

    evaluation = configured.get("evaluation")
    if isinstance(evaluation, dict):
        additional = evaluation.pop("additional_metrics", [])
        secondary = evaluation.get("secondary_metrics", [])
        evaluation["secondary_metrics"] = list(dict.fromkeys([*secondary, *additional]))

    derived = expected_outputs(stage, configured)
    declared = configured.get("expected_outputs")
    if derived is not None and declared is not None and declared != derived:
        raise ConfigError(f"Declared {stage} output counts do not match the derived contract.")
    if derived is not None:
        configured["expected_outputs"] = derived

    validate_stage(configured)
    return configured
