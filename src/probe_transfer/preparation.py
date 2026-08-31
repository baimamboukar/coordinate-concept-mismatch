import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from probe_transfer.artifacts import write_json, write_jsonl
from probe_transfer.atomic import atomic_directory
from probe_transfer.data import (
    assert_disjoint_prepared_splits,
    load_huggingface_dataset,
    prepare_splits,
)
from probe_transfer.extraction.job import ROWS_ENV


def prepare_dataset(config: dict[str, Any], tracker: Any) -> Path:
    output = _output_directory(config)
    dataset = config["dataset"]
    train = dataset["train"]
    test = dataset["test"]
    train_rows = load_huggingface_dataset(
        dataset["id"], dataset["revision"], subset=train["subset"], split=train["split"]
    )
    test_rows = load_huggingface_dataset(
        dataset["id"], dataset["revision"], subset=test["subset"], split=test["split"]
    )
    sampling = config["sampling"]
    clean_test, seeded, audit = prepare_splits(
        train_rows,
        test_rows,
        train_size=sampling["train_size"],
        validation_size=sampling["validation_size"],
        seeds=config["data_seeds"],
        prompt_field=dataset["prompt_field"],
        prompt_template=dataset.get("prompt_template"),
        prompt_fields=dataset.get("prompt_fields"),
        label_field=dataset["label_field"],
        positive_label=dataset["positive_label"],
        negative_label=dataset["negative_label"],
        adversarial_field=dataset["adversarial_field"],
        calibration=sampling.get("disjoint_calibration"),
        fresh_test_size=sampling["test_size"],
    )
    if len(clean_test) != sampling["test_size"]:
        raise ValueError(
            f"Protected test contract expected {sampling['test_size']} rows, found {len(clean_test)}."
        )
    if "partition" in audit:
        assert_disjoint_prepared_splits(
            {
                "test": clean_test,
                **{
                    f"seed_{seed}_{name}": rows
                    for seed, splits in seeded.items()
                    for name, rows in splits.items()
                },
            }
        )
        audit["partition"]["overlap_rows"] = 0

    prepared = {
        "test": clean_test,
        **{
            f"seed_{seed}_{name}": rows
            for seed, splits in seeded.items()
            for name, rows in splits.items()
        },
    }
    if output.exists():
        for name, rows in prepared.items():
            actual = [
                json.loads(line) for line in (output / f"{name}.jsonl").read_text().splitlines()
            ]
            if actual != [_public_row(row) for row in rows]:
                raise ValueError(f"Prepared split differs from the pinned recipe: {name}")
        if "partition" in audit and json.loads((output / "split_audit.json").read_text()) != audit:
            raise ValueError("Prepared split audit differs from the pinned recipe.")
    else:
        with atomic_directory(output) as staging:
            for name, rows in prepared.items():
                write_jsonl(staging / f"{name}.jsonl", map(_public_row, rows))
            if "partition" in audit:
                write_json(staging / "split_audit.json", audit)

    labels = Counter(row["label"] for row in clean_test)
    negative = dataset["negative_label"]
    positive = dataset["positive_label"]
    tracker.report(
        "Data",
        f"Prepared the protected {len(clean_test):,}-row test set "
        f"({labels[0]:,} {negative}, {labels[1]:,} {positive}) and deterministic train/validation "
        f"splits for seeds {config['data_seeds']}. Removed "
        f"{audit['train'].get('duplicate_prompt', 0):,} duplicate train rows.",
    )
    return output


def _output_directory(config: dict[str, Any]) -> Path:
    value = os.getenv(ROWS_ENV) or config.get("preparation", {}).get("output_dir")
    if not value:
        raise RuntimeError(f"{ROWS_ENV} is required for data preparation.")
    return Path(value).expanduser().resolve()


def _public_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: row[key] for key in ("row_id", "prompt", "label", "adversarial")}
