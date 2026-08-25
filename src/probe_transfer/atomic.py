import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import mkdtemp


@contextmanager
def atomic_directory(final: Path) -> Iterator[Path]:
    if final.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {final}")
    final.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(mkdtemp(prefix=f".{final.name}-", dir=final.parent))
    try:
        yield staging
        staging.replace(final)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def publish_directories(staging: Path, destination: Path, names: tuple[str, ...]) -> None:
    targets = [destination / name for name in names]
    if any(target.exists() for target in targets):
        raise FileExistsError("Refusing to overwrite a published output directory.")
    published: list[Path] = []
    try:
        for name, target in zip(names, targets, strict=True):
            (staging / name).replace(target)
            published.append(target)
    except Exception:
        for target in published:
            shutil.rmtree(target, ignore_errors=True)
        raise
