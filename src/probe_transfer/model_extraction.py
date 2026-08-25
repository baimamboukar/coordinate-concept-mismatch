import os
from typing import Any

from probe_transfer.extraction_job import run_extraction_job
from probe_transfer.extraction_types import JobCompletion

MODEL_ENV = "EXTRACTION_MODEL"


def run_model_extraction(config: dict[str, Any], tracker: Any) -> JobCompletion:
    """Run one full model worker selected by ``EXTRACTION_MODEL``."""
    extraction = config.get("extraction", {})
    if config.get("stage") != "modern_baseline" or extraction.get("mode") != "full":
        raise ValueError("Model workers require stage=modern_baseline and extraction.mode=full.")

    model_name = os.getenv(MODEL_ENV)
    if not model_name:
        raise ValueError(f"{MODEL_ENV} must select exactly one model worker.")
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
