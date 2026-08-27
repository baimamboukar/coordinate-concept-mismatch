from typing import Any

from core.config import ConfigError
from probe_transfer.symmetry.protocol import selected_models


def validate_probe_sensitivity(config: dict[str, Any], variants: list[str]) -> None:
    settings = config["symmetry"].get("probe_sensitivity")
    if settings is None:
        return
    families = set(config["probes"]["primary_families"])
    target = settings.get("target_family")
    comparisons = settings.get("comparison_families")
    if target not in families:
        raise ConfigError("Probe-sensitivity target must be a primary probe family.")
    if (
        not isinstance(comparisons, list)
        or len(comparisons) != len(set(comparisons))
        or set(comparisons) != families - {target}
    ):
        raise ConfigError("Probe-sensitivity comparisons must contain every other primary family.")
    selected = settings.get("variants")
    if (
        not isinstance(selected, list)
        or not selected
        or len(selected) != len(set(selected))
        or any(variant not in variants for variant in selected)
    ):
        raise ConfigError("Probe-sensitivity variants must be unique configured scale ranges.")

    target_count = (
        len(config["data_seeds"])
        * len(selected_models(config))
        * len(config["symmetry"]["transformation_seeds"])
    )
    comparison_count = target_count * len(comparisons)
    minimum = settings.get("minimum_target_failures")
    maximum = settings.get("maximum_comparison_failures")
    if type(minimum) is not int or not 1 <= minimum <= target_count:
        raise ConfigError("minimum_target_failures is outside the derived comparison count.")
    if type(maximum) is not int or not 0 <= maximum <= comparison_count:
        raise ConfigError("maximum_comparison_failures is outside the derived comparison count.")
    advantage = settings.get("minimum_mean_gap_advantage")
    reference = settings.get("minimum_reference_auroc")
    if not _nonnegative_number(advantage):
        raise ConfigError("minimum_mean_gap_advantage must be non-negative.")
    if not _unit_number(reference):
        raise ConfigError("minimum_reference_auroc must lie in [0, 1].")


def _nonnegative_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0


def _unit_number(value: Any) -> bool:
    return _nonnegative_number(value) and value <= 1
