from typing import Any

from probe_transfer.symmetry.protocol import estimated_alignment_enabled, selected_models


def expected_outputs(stage: str, config: dict[str, Any]) -> dict[str, int] | None:
    builders = {
        "transfer": _transfer,
        "align": _alignment,
        "symmetry": _symmetry,
    }
    builder = builders.get(stage)
    return None if builder is None else builder(config)


def _families(config: dict[str, Any], depths: list[float], primary_depth: float) -> int:
    probes = config["probes"]
    return sum(
        len(probes["primary_families"])
        if depth == primary_depth
        else len(probes["secondary_families"])
        for depth in depths
    )


def _transfer(config: dict[str, Any]) -> dict[str, int]:
    models = len(config["models"])
    seeds = len(config["data_seeds"])
    activations = config["activations"]
    family_depths = _families(
        config,
        activations["normalized_depths"],
        activations["primary_depth"],
    )
    evaluations = seeds * family_depths * models**2
    comparisons = seeds * family_depths * models * (models - 1)
    return {
        "metrics_rows": evaluations,
        "prediction_rows": evaluations * config["sampling"]["test_size"],
        "transfer_gap_rows": comparisons,
        "probe_bundles": seeds * models,
    }


def _alignment(config: dict[str, Any]) -> dict[str, int]:
    alignment = config["alignment"]
    probes = config["probes"]
    directions = sum(len(pairs) for pairs in config["evaluation"]["pair_groups"].values())
    metrics_per_seed = 0
    recoveries_per_seed = 0
    ambient = len([name for name in alignment["methods"] if name != "quotient_ridge"]) + 1
    quotient = "quotient_ridge" in alignment["methods"]
    for depth in alignment["depths"]:
        families = (
            probes["primary_families"]
            if depth == alignment["primary_depth"]
            else probes["secondary_families"]
        )
        for family in families:
            methods = ambient + int(quotient and family == "linear")
            metrics_per_seed += 2 + methods
            recoveries_per_seed += methods

    seeds = len(config["data_seeds"])
    metrics = seeds * directions * metrics_per_seed
    diagnostics = seeds * directions * len(alignment["depths"]) * (ambient + int(quotient))
    return {
        "metrics_rows": metrics,
        "prediction_rows": metrics * config["materials"]["expected_test_rows"],
        "recovery_rows": seeds * directions * recoveries_per_seed,
        "alignment_diagnostic_rows": diagnostics,
    }


def _symmetry(config: dict[str, Any]) -> dict[str, int]:
    symmetry = config["symmetry"]
    seeds = len(config["data_seeds"])
    models = len(selected_models(config))
    family_depths = _families(
        config,
        symmetry["probed_depths"],
        symmetry["primary_depth"],
    )
    permutations = len(symmetry["permutation_seeds"])
    comparisons = seeds * models * family_depths
    estimated = estimated_alignment_enabled(config)
    conditions = 2 + (3 + int(estimated)) * permutations
    outputs = {
        "metrics_rows": comparisons * conditions,
        "prediction_rows": comparisons * conditions * config["materials"]["expected_test_rows"],
        "recovery_rows": comparisons * permutations,
        "function_gate_rows": models * (1 + permutations),
        "probe_bundles": seeds * models * permutations,
    }
    smoke_rows = symmetry.get("smoke_gate_rows", 0)
    if smoke_rows:
        outputs["function_smoke_gate_rows"] = models * (1 + permutations)
    if estimated:
        outputs["alignment_diagnostic_rows"] = (
            seeds * models * len(symmetry["probed_depths"]) * permutations
        )
    return outputs
