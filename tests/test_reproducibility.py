import pytest

from core.reproducibility import require_process_hash_seed, seed_everything


def test_seed_everything_repeats_python_randomness() -> None:
    import random

    seed_everything(42)
    first = random.random()
    seed_everything(42)

    assert random.random() == first


def test_process_hash_seed_must_be_set_before_start(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    with pytest.raises(RuntimeError, match="PYTHONHASHSEED=42"):
        require_process_hash_seed(42)

    monkeypatch.setenv("PYTHONHASHSEED", "42")
    require_process_hash_seed(42)
