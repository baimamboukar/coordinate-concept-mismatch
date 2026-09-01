from pathlib import Path
from typing import Any

import numpy as np

from probe_transfer.alignment import cross_task
from probe_transfer.alignment.materials import paired_split
from probe_transfer.alignment.methods import AlignmentMap
from probe_transfer.alignment.selection import fit_configured_alignments
from probe_transfer.alignment.task_adaptation import fit_task_adaptations


def fit_evaluation_maps(
    baseline_dir: Path,
    fit_root: Path,
    config: dict[str, Any],
    *,
    source: str,
    target: str,
    data_seed: int,
    depth: float,
    layer: str,
    device: str,
) -> tuple[dict[str, AlignmentMap], list[dict[str, Any]], tuple[np.ndarray, np.ndarray]]:
    alignment = config["alignment"]
    split = alignment.get("fit_split", "train")
    train = cross_task.load_fit_split(
        fit_root, config, source, target, f"seed_{data_seed}_{split}", layer
    )
    _assert_rows(train, cross_task.fit_expected_rows(config), "shared-map fit")
    maps, records = fit_configured_alignments(
        train[0],
        train[1],
        config,
        fit_root,
        source=source,
        target=target,
        data_seed=data_seed,
        layer=layer,
        shuffle_seed=alignment["shuffled_pairing_seed"],
        device=device,
    )
    settings = alignment.get("task_adaptation")
    if settings is not None:
        task_split = settings["calibration_split"]
        calibration = paired_split(
            baseline_dir, source, target, f"seed_{data_seed}_{task_split}", layer
        )
        _assert_rows(
            calibration,
            config["materials"][f"expected_{task_split}_rows"],
            "task calibration",
        )
        base = maps[settings["base_method"]]
        shuffle_seed = (
            alignment["shuffled_pairing_seed"]
            + 10_000 * data_seed
            + sorted(config["models"]).index(source)
        )
        maps.update(
            fit_task_adaptations(
                base,
                calibration[0],
                calibration[1],
                settings,
                shuffle_seed=shuffle_seed,
                device=device,
            )
        )
    context = {
        "data_seed": data_seed,
        "source_model": source,
        "target_model": target,
        "depth": depth,
    }
    return maps, [{**context, **record} for record in records], train[:2]


def _assert_rows(values: tuple, expected: int, label: str) -> None:
    if any(len(value) != expected for value in values):
        raise ValueError(f"Expected {expected} paired rows for {label}.")
