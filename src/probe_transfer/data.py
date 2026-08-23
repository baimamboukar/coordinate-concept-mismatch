import hashlib
import json
import os
import random
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, TypedDict

from core.reproducibility import is_pinned_hf_revision


class CleanRowFields(TypedDict):
    prompt_field: str
    label_field: str
    positive_label: str
    negative_label: str
    adversarial_field: str


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


def load_prepared_rows(path: str | Path, expected_size: int) -> list[dict[str, Any]]:
    if expected_size < 2 or expected_size % 2:
        raise ValueError("Expected prepared row count must be positive and even.")

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
    if sum(row["label"] for row in rows) * 2 != expected_size:
        raise ValueError("Prepared rows must be label-balanced.")
    return rows


def normalize_prompt(prompt: str) -> str:
    normalized = unicodedata.normalize("NFKC", prompt)
    return " ".join(normalized.split()).casefold()


def clean_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    prompt_field: str,
    label_field: str,
    positive_label: str,
    negative_label: str,
    adversarial_field: str,
    reserved_digests: set[str] | None = None,
    audit: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    labels = {negative_label: 0, positive_label: 1}
    reserved = set(reserved_digests or ())
    groups: dict[str, dict[str, Any]] = {}
    counts = audit if audit is not None else {}

    for row_id, row in enumerate(rows):
        counts["input_rows"] = counts.get("input_rows", 0) + 1
        prompt = row.get(prompt_field)
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
                "adversarial": row.get(adversarial_field),
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
    prompt_field: str,
    label_field: str,
    positive_label: str,
    negative_label: str,
    adversarial_field: str,
) -> tuple[
    list[dict[str, Any]],
    dict[int, dict[str, list[dict[str, Any]]]],
    dict[str, dict[str, int]],
]:
    common: CleanRowFields = {
        "prompt_field": prompt_field,
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
        selected_train = _balanced_sample(train_pool, train_size, rng)
        selected_ids = {row["prompt_sha256"] for row in selected_train}
        remaining = [row for row in train_pool if row["prompt_sha256"] not in selected_ids]
        validation = _balanced_sample(remaining, validation_size, rng)
        seed_splits[seed] = {"train": selected_train, "validation": validation}
    return test, seed_splits, audit


def balanced_subset(rows: list[dict[str, Any]], size: int, seed: int) -> list[dict[str, Any]]:
    return _balanced_sample(rows, size, random.Random(seed))


def _balanced_sample(
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
