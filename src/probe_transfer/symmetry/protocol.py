from typing import Any


def selected_models(config: dict[str, Any]) -> list[str]:
    worker = config.get("worker_model")
    if worker is not None:
        return [worker]
    configured = config["symmetry"].get("models")
    return list(configured) if configured is not None else list(config["models"])


def estimated_alignment_enabled(config: dict[str, Any]) -> bool:
    return bool(config["symmetry"].get("estimated_alignment", {}).get("enabled", False))


def transformation_count(config: dict[str, Any]) -> int:
    symmetry = config["symmetry"]
    ranges = symmetry.get("scale_ranges")
    variants = len(ranges) if isinstance(ranges, dict) else 1
    return len(symmetry["transformation_seeds"]) * variants
