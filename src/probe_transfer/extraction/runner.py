from typing import Any

from probe_transfer.extraction.job import run_extraction_job
from probe_transfer.extraction.types import JobCompletion


def run_model_extraction(
    config: dict[str, Any], tracker: Any, model_name: str | None = None
) -> JobCompletion:
    """Run one full model extraction worker."""
    extraction = config.get("extraction", {})
    if config.get("stage") != "extract" or extraction.get("mode") != "full":
        raise ValueError("Model workers require the configured full extraction stage.")
    if not model_name:
        raise ValueError("Select exactly one model worker.")
    allowed = extraction.get("models", ())
    if model_name not in allowed:
        raise ValueError(f"Model {model_name} is not enabled by extraction.models.")

    completion = run_extraction_job(config, model_name=model_name)
    total_rows = sum(split.rows for split in completion.splits)
    total_truncated = sum(split.truncated_rows for split in completion.splits)
    tracker.metrics(
        {
            "extraction/rows": float(total_rows),
            "extraction/truncation_rate": total_truncated / total_rows,
        }
    )
    return completion
