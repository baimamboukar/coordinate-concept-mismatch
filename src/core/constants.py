from pathlib import Path

PROJECT_NAME = "coordinate-concept-mismatch"
PAPER_TITLE = "Coordinate or Concept Mismatch? Disentangling Cross-Model Probe Transfer Failure"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = PROJECT_ROOT / "configs"
LOGS_DIR = PROJECT_ROOT / "logs"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

CONFIG_SUFFIX = ".yaml"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
EXPERIMENT_NAME_PATTERN = r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)+$"
GENERIC_EXPERIMENT_PATTERN = r"^(?:exp|experiment)_\d+$"
HF_COMMIT_PATTERN = r"^[0-9a-f]{40}$"

REQUIRED_BINARY_METRICS = frozenset(
    {
        "accuracy",
        "auprc",
        "balanced_accuracy",
        "f1",
        "fn",
        "fp",
        "precision",
        "recall",
        "tn",
        "tp",
        "tpr_at_fpr",
    }
)
REQUIRED_ROW_LEVEL_FIELDS = frozenset({"row_id", "label", "score", "probability", "prediction"})
