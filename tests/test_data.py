import pytest

from probe_transfer.data import load_huggingface_dataset, normalize_prompt, prepare_splits


def test_huggingface_dataset_revision_must_be_pinned() -> None:
    with pytest.raises(ValueError, match="40-character commit"):
        load_huggingface_dataset("organization/dataset", "main")


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
