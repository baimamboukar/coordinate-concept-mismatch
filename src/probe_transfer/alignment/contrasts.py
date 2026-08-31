import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from core.config import ConfigError
from probe_transfer.probes.evaluation import paired_auroc_gap_interval

CONTEXT_FIELDS = (
    "data_seed",
    "depth",
    "probe_family",
    "source_model",
    "target_model",
    "pair_group",
)


def validate_contrasts(study: dict[str, Any]) -> None:
    settings = study.get("decision_rules", {}).get("contrasts", {})
    if not isinstance(settings, dict):
        raise ConfigError("Condition contrasts must be a mapping.")
    enabled = {name for name, value in study.get("fit_conditions", {}).items() if value is not None}
    for contrast in settings.values():
        if not isinstance(contrast, dict) or set(contrast) != {"reference", "treatment", "primary"}:
            raise ConfigError("Each contrast requires reference, treatment, and primary fields.")
        if contrast["reference"] not in enabled or contrast["treatment"] not in enabled:
            raise ConfigError("Contrast conditions must be enabled fitting conditions.")
        if contrast["reference"] == contrast["treatment"] or type(contrast["primary"]) is not bool:
            raise ConfigError("Contrasts require distinct conditions and a boolean primary flag.")
    if settings and sum(row["primary"] for row in settings.values()) != 1:
        raise ConfigError("Specify exactly one primary condition contrast.")


def condition_contrasts(
    root: Path, study: dict[str, Any], tasks: list[str]
) -> list[dict[str, Any]]:
    validate_contrasts(study)
    rows = []
    for task in tasks:
        for name, spec in study.get("decision_rules", {}).get("contrasts", {}).items():
            paths = [root / task / spec[key] / "results" for key in ("reference", "treatment")]
            reference, treatment = [_scores(path / "predictions.jsonl") for path in paths]
            recoveries = [_recovery(path / "recovery.jsonl") for path in paths]
            if reference.keys() != treatment.keys():
                raise ValueError("Condition contrasts require identical comparison contexts.")
            if any(lookup.keys() != reference.keys() for lookup in recoveries):
                raise ValueError("Contrast recovery records must match the prediction contexts.")
            for context in sorted(reference):
                before, after = reference[context], treatment[context]
                ids = sorted(before)
                if before.keys() != after.keys() or any(before[i][0] != after[i][0] for i in ids):
                    raise ValueError("Condition contrasts require identical row IDs and labels.")
                evaluation = study["evaluation"]
                delta, lower, upper = paired_auroc_gap_interval(
                    np.array([before[i][0] for i in ids]),
                    np.array([after[i][1] for i in ids]),
                    np.array([before[i][1] for i in ids]),
                    samples=evaluation["bootstrap_samples"],
                    confidence=evaluation["confidence_level"],
                    seed=study["seed"] + len(rows),
                )
                old, new = [lookup[context] for lookup in recoveries]
                if abs(old["raw_auroc_gap"] - new["raw_auroc_gap"]) > 1e-8:
                    raise ValueError("Condition contrasts must share a frozen-transfer baseline.")
                before_recovery, after_recovery = old["recovery_fraction"], new["recovery_fraction"]
                rows.append(
                    {
                        "task": task,
                        "contrast": name,
                        **spec,
                        **dict(zip(CONTEXT_FIELDS, context, strict=True)),
                        "auroc_change": delta,
                        "ci_lower": lower,
                        "ci_upper": upper,
                        "recovery_change": None
                        if before_recovery is None or after_recovery is None
                        else after_recovery - before_recovery,
                    }
                )
    return rows


def _scores(path: Path):
    groups = defaultdict(dict)
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["condition"] != "affine_ridge":
                continue
            context = tuple(row[field] for field in CONTEXT_FIELDS)
            if row["row_id"] in groups[context]:
                raise ValueError("Contrast predictions contain duplicate row IDs.")
            groups[context][row["row_id"]] = (row["label"], row["score"])
    if not groups:
        raise ValueError("Condition contrasts require affine-ridge predictions.")
    return groups


def _recovery(path: Path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    selected = [row for row in rows if row["method"] == "affine_ridge"]
    indexed = {tuple(row[field] for field in CONTEXT_FIELDS): row for row in selected}
    if len(indexed) != len(selected):
        raise ValueError("Contrast recovery records contain duplicate contexts.")
    return indexed
