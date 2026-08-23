from core.reproducibility import seed_everything


def test_seed_everything_repeats_python_randomness() -> None:
    import random

    seed_everything(42)
    first = random.random()
    seed_everything(42)

    assert random.random() == first
