import hashlib
import json
import os
import random
import unicodedata
from collections.abc import Iterable, Mapping
from pathlib import Path
from string import Formatter
from typing import Any, TypedDict

from core.config import ConfigError
from core.reproducibility import is_pinned_hf_revision
from probe_transfer.splits import add_disjoint_calibration, balanced_sample


class CleanRowFields(TypedDict):
    prompt_field: str | None
    prompt_template: str | None
    prompt_fields: tuple[str, ...]
    label_field: str
    positive_label: Any
    negative_label: Any
    adversarial_field: str | None


def load_huggingface_dataset(
    dataset_id: str,
    revision: str,
    *,
    subset: str | None = None,
    split: str | None = None,
    **parameters: Any,
) -> Any:
    if not is_pinned_hf_revision(revision):
        raise ValueError("Hugging Face datasets require an exact 40-character commit revision.")

    from datasets import load_dataset

    return load_dataset(
        dataset_id,
        subset,
        split=split,
        revision=revision,
        token=os.getenv("HF_TOKEN"),
        **parameters,
    )


def load_prepared_rows(
    path: str | Path,
    expected_size: int,
    *,
    require_balanced: bool = True,
) -> list[dict[str, Any]]:
    if expected_size < 2 or (require_balanced and expected_size % 2):
        requirement = "positive and even" if require_balanced else "at least two"
        raise ValueError(f"Expected prepared row count must be {requirement}.")

    rows = []
    for line_number, line in enumerate(Path(path).read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise TypeError(f"Prepared row {line_number} must be a JSON object.")
        if type(row.get("row_id")) is not int:
            raise ValueError(f"Prepared row {line_number} requires an integer row_id.")
        if not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
            raise ValueError(f"Prepared row {line_number} requires a non-empty prompt.")
        if type(row.get("label")) is not int or row["label"] not in (0, 1):
            raise ValueError(f"Prepared row {line_number} requires a binary integer label.")
        rows.append(row)

    if len(rows) != expected_size:
        raise ValueError(f"Expected {expected_size} prepared rows, received {len(rows)}.")
    if len({row["row_id"] for row in rows}) != expected_size:
        raise ValueError("Prepared row IDs must be unique.")
    labels = [row["label"] for row in rows]
    if set(labels) != {0, 1}:
        raise ValueError("Prepared rows must contain both binary labels.")
    if require_balanced and sum(labels) * 2 != expected_size:
        raise ValueError("Prepared rows must be label-balanced.")
    return rows


def normalize_prompt(prompt: str) -> str:
    normalized = unicodedata.normalize("NFKC", prompt)
    return " ".join(normalized.split()).casefold()


def validate_prompt_configuration(dataset: Mapping[str, Any]) -> None:
    field = dataset.get("prompt_field")
    template = dataset.get("prompt_template")
    fields = dataset.get("prompt_fields")
    if template is None:
        if not isinstance(field, str) or not field:
            raise ConfigError("Datasets require a prompt_field or prompt_template.")
        return
    if field is not None or not isinstance(template, str) or not template.strip():
        raise ConfigError("Templated prompts require prompt_field: null and a non-empty template.")
    if (
        not isinstance(fields, list)
        or not fields
        or any(not isinstance(name, str) or not name for name in fields)
        or len(fields) != len(set(fields))
    ):
        raise ConfigError("Templated prompts require unique non-empty prompt_fields.")
    placeholders = {name for _, name, _, _ in Formatter().parse(template) if name is not None}
    if placeholders != set(fields):
        raise ConfigError("Prompt template placeholders must exactly match prompt_fields.")


def clean_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    prompt_field: str | None,
    label_field: str,
    positive_label: Any,
    negative_label: Any,
    adversarial_field: str | None,
    prompt_template: str | None = None,
    prompt_fields: tuple[str, ...] = (),
    reserved_digests: set[str] | None = None,
    audit: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    labels = {negative_label: 0, positive_label: 1}
    reserved = set(reserved_digests or ())
    groups: dict[str, dict[str, Any]] = {}
    counts = audit if audit is not None else {}

    for row_id, row in enumerate(rows):
        counts["input_rows"] = counts.get("input_rows", 0) + 1
        prompt = _render_prompt(row, prompt_field, prompt_template, prompt_fields)
        label = row.get(label_field)
        if not isinstance(prompt, str) or not prompt.strip():
            counts["invalid_prompt"] = counts.get("invalid_prompt", 0) + 1
            continue
        if label not in labels:
            counts["invalid_label"] = counts.get("invalid_label", 0) + 1
            continue

        digest = hashlib.sha256(normalize_prompt(prompt).encode()).hexdigest()
        if digest in reserved:
            counts["protected_test_overlap"] = counts.get("protected_test_overlap", 0) + 1
            continue
        encoded_label = labels[label]
        if digest in groups:
            groups[digest]["count"] += 1
            groups[digest]["labels"].add(encoded_label)
            continue
        groups[digest] = {
            "count": 1,
            "labels": {encoded_label},
            "record": {
                "row_id": row_id,
                "prompt": prompt,
                "prompt_sha256": digest,
                "label": encoded_label,
                "adversarial": row.get(adversarial_field) if adversarial_field else None,
            },
        }

    cleaned = []
    for group in groups.values():
        if len(group["labels"]) > 1:
            counts["conflicting_label_prompts"] = counts.get("conflicting_label_prompts", 0) + 1
            counts["conflicting_label_rows"] = (
                counts.get("conflicting_label_rows", 0) + group["count"]
            )
            continue
        counts["duplicate_prompt"] = counts.get("duplicate_prompt", 0) + group["count"] - 1
        cleaned.append(group["record"])
    counts["output_rows"] = len(cleaned)
    return cleaned


def prepare_splits(
    train_rows: Iterable[Mapping[str, Any]],
    test_rows: Iterable[Mapping[str, Any]],
    *,
    train_size: int,
    validation_size: int,
    seeds: list[int],
    prompt_field: str | None,
    label_field: str,
    positive_label: Any,
    negative_label: Any,
    adversarial_field: str | None,
    prompt_template: str | None = None,
    prompt_fields: list[str] | None = None,
    calibration: Mapping[str, Any] | None = None,
    fresh_test_size: int | None = None,
) -> tuple[
    list[dict[str, Any]],
    dict[int, dict[str, list[dict[str, Any]]]],
    dict[str, dict[str, int]],
]:
    common: CleanRowFields = {
        "prompt_field": prompt_field,
        "prompt_template": prompt_template,
        "prompt_fields": tuple(prompt_fields or ()),
        "label_field": label_field,
        "positive_label": positive_label,
        "negative_label": negative_label,
        "adversarial_field": adversarial_field,
    }
    audit = {"train": {}, "test": {}}
    test = clean_rows(test_rows, audit=audit["test"], **common)
    protected = {row["prompt_sha256"] for row in test}
    train_pool = clean_rows(
        train_rows,
        reserved_digests=protected,
        audit=audit["train"],
        **common,
    )

    seed_splits = {}
    for seed in seeds:
        rng = random.Random(seed)
        selected_train = balanced_sample(train_pool, train_size, rng)
        selected_ids = {row["prompt_sha256"] for row in selected_train}
        remaining = [row for row in train_pool if row["prompt_sha256"] not in selected_ids]
        validation = balanced_sample(remaining, validation_size, rng)
        seed_splits[seed] = {"train": selected_train, "validation": validation}
    if calibration is not None:
        if fresh_test_size is None:
            raise ValueError("Disjoint calibration requires an explicit fresh test size.")
        test, seed_splits, audit["partition"] = add_disjoint_calibration(
            train_pool, seed_splits, settings=calibration, test_size=fresh_test_size
        )
    return test, seed_splits, audit


def _render_prompt(
    row: Mapping[str, Any],
    field: str | None,
    template: str | None,
    fields: tuple[str, ...],
) -> Any:
    if template is None:
        return row.get(field) if field is not None else None
    values = {name: row.get(name) for name in fields}
    if any(not isinstance(value, str) or not value.strip() for value in values.values()):
        return None
    return template.format_map(values)


def balanced_subset(rows: list[dict[str, Any]], size: int, seed: int) -> list[dict[str, Any]]:
    return balanced_sample(rows, size, random.Random(seed))


def assert_disjoint_prepared_splits(splits: Mapping[str, list[dict[str, Any]]]) -> None:
    if not any(name.endswith("_calibration") for name in splits):
        return
    keys = {
        name: {normalize_prompt(row["prompt"]) for row in rows} for name, rows in splits.items()
    }
    if any(len(keys[name]) != len(rows) for name, rows in splits.items()):
        raise ValueError("Disjoint partitions contain duplicate normalized prompts.")
    probe_names = [name for name in keys if "calibration" not in name and name != "test"]
    used = set().union(*(keys[name] for name in probe_names))
    fresh = [name for name in keys if "calibration" in name]
    if any(keys[name] & used for name in ["test", *fresh]):
        raise ValueError("Fresh calibration or test rows overlap prior probe splits.")
    if any(keys["test"] & keys[name] for name in fresh):
        raise ValueError("Fresh test rows overlap calibration splits.")
    for name in fresh:
        if name.endswith("_calibration") and keys[name] & keys[f"{name}_validation"]:
            raise ValueError("Calibration fitting and validation rows overlap.")
