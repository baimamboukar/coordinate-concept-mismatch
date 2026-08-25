import os
import random
import re
from typing import Any

from core.constants import HF_COMMIT_PATTERN


def is_pinned_hf_revision(revision: Any) -> bool:
    return isinstance(revision, str) and re.fullmatch(HF_COMMIT_PATTERN, revision) is not None


def seed_everything(seed: int, deterministic: bool = True) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ModuleNotFoundError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True)
            torch.set_float32_matmul_precision("highest")
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
    except ModuleNotFoundError:
        pass


def require_process_hash_seed(seed: int) -> None:
    if os.getenv("PYTHONHASHSEED") != str(seed):
        raise RuntimeError(f"Start Python with PYTHONHASHSEED={seed} for deterministic execution.")
