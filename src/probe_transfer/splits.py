import random
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

from core.config import ConfigError


def seeded_split_sizes(config: Mapping[str, Any]) -> dict[str, int]:
    sampling = config["sampling"]
    sizes = {split: sampling[f"{split}_size"] for split in ("train", "validation")}
    calibration = sampling.get("disjoint_calibration")
    if calibration is not None:
        sizes.update(
            calibration=calibration["train_size"],
            calibration_validation=calibration["validation_size"],
        )
    return sizes


def validate_split_configuration(config: Mapping[str, Any]) -> None:
    sampling = config["sampling"]
    settings = sampling.get("disjoint_calibration")
    if settings is None:
        return
    required = {"train_size", "validation_size", "holdout_seed", "evaluation_source"}
    if not isinstance(settings, dict) or set(settings) != required:
        raise ConfigError(
            "Disjoint calibration requires sizes, holdout_seed, and evaluation_source."
        )
    if settings["evaluation_source"] != "unused_training_pool":
        raise ConfigError("Fresh evaluation must use the unused_training_pool.")
    if type(settings["holdout_seed"]) is not int or settings["holdout_seed"] < 0:
        raise ConfigError("The fresh holdout seed must be a non-negative integer.")
    for size in (*seeded_split_sizes(config).values(), sampling["test_size"]):
        if type(size) is not int or size < 2 or size % 2:
            raise ConfigError("Disjoint split sizes must be positive even integers.")


def add_disjoint_calibration(pool, seeded, *, settings, test_size):
    used = {
        row["prompt_sha256"]
        for splits in seeded.values()
        for rows in splits.values()
        for row in rows
    }
    unused = [row for row in pool if row["prompt_sha256"] not in used]
    required = test_size + settings["train_size"] + settings["validation_size"]
    if any(sum(row["label"] == label for row in unused) < required // 2 for label in (0, 1)):
        raise ValueError("Insufficient unused rows for the fixed calibration and holdout budgets.")
    test = balanced_sample(unused, test_size, random.Random(settings["holdout_seed"]))
    protected = {row["prompt_sha256"] for row in test}
    remaining = [row for row in unused if row["prompt_sha256"] not in protected]
    result = {}
    for seed, splits in seeded.items():
        rng = random.Random(seed)
        calibration = balanced_sample(remaining, settings["train_size"], rng)
        selected = {row["prompt_sha256"] for row in calibration}
        validation_pool = [row for row in remaining if row["prompt_sha256"] not in selected]
        validation = balanced_sample(validation_pool, settings["validation_size"], rng)
        result[seed] = {**splits, "calibration": calibration, "calibration_validation": validation}
    audit = {
        "prior_probe_train_validation_union": len(used),
        "unused_pool": len(unused),
        "fresh_test_rows": len(test),
        "calibration_rows_per_seed": settings["train_size"],
        "calibration_validation_rows_per_seed": settings["validation_size"],
    }
    return test, result, audit


def balanced_sample(
    rows: list[dict[str, Any]], size: int, rng: random.Random
) -> list[dict[str, Any]]:
    if size <= 0 or size % 2:
        raise ValueError("Balanced sample sizes must be positive even integers.")
    selected = []
    for label in (0, 1):
        bucket = [row for row in rows if row["label"] == label]
        selected.extend(_sample_adversarial_strata(bucket, size // 2, rng))
    rng.shuffle(selected)
    return selected


def _sample_adversarial_strata(
    rows: list[dict[str, Any]], size: int, rng: random.Random
) -> list[dict[str, Any]]:
    if len(rows) < size:
        raise ValueError(f"Requested {size} rows from a stratum containing {len(rows)}.")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["adversarial"])].append(row)
    exact = {key: size * len(group) / len(rows) for key, group in groups.items()}
    quotas = {key: int(value) for key, value in exact.items()}
    remainder = size - sum(quotas.values())
    order = sorted(groups, key=lambda key: (exact[key] - quotas[key], key), reverse=True)
    for key in order[:remainder]:
        quotas[key] += 1
    selected = []
    for key in sorted(groups):
        selected.extend(rng.sample(groups[key], quotas[key]))
    return selected
