from dataclasses import dataclass


@dataclass(frozen=True)
class PreparedSplit:
    name: str
    expected_rows: int
    balanced: bool
    data_seed: int | None = None

    @property
    def input_name(self) -> str:
        return f"{self.name}.jsonl"

    @property
    def output_name(self) -> str:
        return f"{self.name}.safetensors"


@dataclass(frozen=True)
class SplitCompletion:
    split: str
    data_seed: int | None
    path: str
    rows: int
    truncated_rows: int
    truncation_rate: float


@dataclass(frozen=True)
class JobCompletion:
    schema_version: int
    status: str
    model_name: str
    model_id: str
    model_revision: str
    block_indices: tuple[int, ...]
    normalized_depths: tuple[float, ...]
    splits: tuple[SplitCompletion, ...]
