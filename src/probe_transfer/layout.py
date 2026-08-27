import re
from pathlib import PurePosixPath
from typing import Any

SEMANTIC_KEY = re.compile(r"^[a-z0-9]+(?:-[a-z0-9.]+)*$")


def activation_prefix(config: dict[str, Any], model_name: str) -> str:
    prefix = (
        f"activations/{_key(config['artifacts']['dataset_key'])}/"
        f"{model_artifact_key(config, model_name)}"
    )
    variant = config.get("activations", {}).get("artifact_key")
    return prefix if variant is None else f"{prefix}/{_key(variant)}"


def model_artifact_key(config: dict[str, Any], model_name: str) -> str:
    model = config["models"][model_name]
    return _key(model.get("artifact_key", model_name.replace("_", "-")))


def stage_prefix(config: dict[str, Any]) -> str:
    return study_prefix(config["name"], config["study"], config.get("artifact_variant"))


def study_prefix(objective: str, study: str, variant: str | None = None) -> str:
    objective_key = _key(objective.replace("_", "-"))
    study_key = _key(study.replace("_", "-"))
    prefix = f"studies/{objective_key}/{study_key}"
    return prefix if variant is None else f"{prefix}/{_key(variant)}"


def artifact_uri(config: dict[str, Any], prefix: str) -> str:
    return bucket_uri(config["artifacts"]["bucket"], prefix)


def bucket_uri(bucket: str, remote_path: str) -> str:
    path = PurePosixPath(remote_path)
    if path.is_absolute() or ".." in path.parts or not path.name:
        raise ValueError("Bucket paths must be non-empty relative paths without '..'.")
    if bucket.count("/") != 1:
        raise ValueError("Bucket IDs must use the form 'namespace/name'.")
    return f"hf://buckets/{bucket}/{path}"


def _key(value: str) -> str:
    if not SEMANTIC_KEY.fullmatch(value):
        raise ValueError(f"Artifact keys must be semantic lowercase slugs: {value}")
    return value
