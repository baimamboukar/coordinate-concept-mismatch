import json
from pathlib import Path

import pytest

from probe_transfer.data import (
    load_huggingface_dataset,
    load_prepared_rows,
    normalize_prompt,
    prepare_splits,
    validate_prompt_configuration,
)


def test_huggingface_dataset_revision_must_be_pinned() -> None:
    with pytest.raises(ValueError, match="40-character commit"):
        load_huggingface_dataset("organization/dataset", "main")


def test_loads_validated_prepared_rows(tmp_path: Path) -> None:
    rows = [
        {"row_id": 1, "prompt": "safe", "label": 0, "adversarial": False},
        {"row_id": 2, "prompt": "unsafe", "label": 1, "adversarial": True},
    ]
    path = tmp_path / "rows.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows))

    assert load_prepared_rows(path, 2) == rows
    with pytest.raises(ValueError, match="Expected 4"):
        load_prepared_rows(path, 4)


def test_loads_unbalanced_protected_test_rows(tmp_path: Path) -> None:
    rows = [
        {"row_id": 1, "prompt": "safe one", "label": 0},
        {"row_id": 2, "prompt": "safe two", "label": 0},
        {"row_id": 3, "prompt": "unsafe", "label": 1},
    ]
    path = tmp_path / "test.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows))

    assert len(load_prepared_rows(path, 3, require_balanced=False)) == 3


def test_normalize_prompt_collapses_equivalent_text() -> None:
    assert normalize_prompt("  Harmful\nPrompt ") == normalize_prompt("harmful prompt")


def test_splits_are_balanced_deterministic_and_test_protected() -> None:
    train = [
        {
            "prompt": f"prompt {label} {index}",
            "prompt_harm_label": label,
            "adversarial": index % 2 == 0,
        }
        for label in ("harmful", "unharmful")
        for index in range(12)
    ]
    train.append({"prompt": "protected", "prompt_harm_label": "harmful", "adversarial": False})
    train.extend(
        [
            {"prompt": "conflict", "prompt_harm_label": "harmful", "adversarial": False},
            {"prompt": "conflict", "prompt_harm_label": "unharmful", "adversarial": False},
        ]
    )
    test = [
        {"prompt": "protected", "prompt_harm_label": "harmful", "adversarial": False},
        {"prompt": "safe test", "prompt_harm_label": "unharmful", "adversarial": True},
    ]
    parameters = {
        "train_size": 12,
        "validation_size": 4,
        "seeds": [42],
        "prompt_field": "prompt",
        "label_field": "prompt_harm_label",
        "positive_label": "harmful",
        "negative_label": "unharmful",
        "adversarial_field": "adversarial",
    }

    cleaned_test, first, audit = prepare_splits(train, test, **parameters)
    _, second, _ = prepare_splits(train, test, **parameters)

    assert first == second
    assert {row["label"] for row in cleaned_test} == {0, 1}
    assert [row["label"] for row in first[42]["train"]].count(1) == 6
    assert all(row["prompt"] != "protected" for row in first[42]["train"])
    assert all(row["prompt"] != "conflict" for row in first[42]["train"])
    assert audit["train"]["protected_test_overlap"] == 1
    assert audit["train"]["conflicting_label_prompts"] == 1


def test_splits_accept_numeric_labels_without_a_stratification_field() -> None:
    train = [
        {"sentence": f"sentiment {label} {index}", "label": label}
        for label in (0, 1)
        for index in range(12)
    ]
    test = [
        {"sentence": "negative test", "label": 0},
        {"sentence": "positive test", "label": 1},
    ]

    protected, seeded, _ = prepare_splits(
        train,
        test,
        train_size=12,
        validation_size=4,
        seeds=[42],
        prompt_field="sentence",
        label_field="label",
        positive_label=1,
        negative_label=0,
        adversarial_field=None,
    )

    assert {row["label"] for row in protected} == {0, 1}
    assert sum(row["label"] for row in seeded[42]["train"]) == 6
    assert all(row["adversarial"] is None for row in seeded[42]["train"])


def test_splits_render_a_composed_prompt() -> None:
    rows = [
        {"question": f"question {label} {index}", "passage": "evidence", "answer": label}
        for label in (False, True)
        for index in range(8)
    ]
    protected, seeded, _ = prepare_splits(
        rows,
        rows[:2],
        train_size=8,
        validation_size=2,
        seeds=[42],
        prompt_field=None,
        prompt_template="Question: {question}\nPassage: {passage}",
        prompt_fields=["question", "passage"],
        label_field="answer",
        positive_label=True,
        negative_label=False,
        adversarial_field=None,
    )

    assert protected[0]["prompt"].startswith("Question: ")
    assert all("\nPassage: evidence" in row["prompt"] for row in seeded[42]["train"])


def test_prompt_template_fields_must_match() -> None:
    with pytest.raises(ValueError, match="exactly match"):
        validate_prompt_configuration(
            {
                "prompt_field": None,
                "prompt_template": "Question: {question}",
                "prompt_fields": ["question", "passage"],
            }
        )
