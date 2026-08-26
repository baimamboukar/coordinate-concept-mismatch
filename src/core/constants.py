from pathlib import Path

PROJECT_NAME = "coordinate-concept-mismatch"
PAPER_TITLE = "Coordinate or Concept Mismatch? Disentangling Cross-Model Probe Transfer Failure"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = PROJECT_ROOT / "configs"
LOGS_DIR = PROJECT_ROOT / "logs"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
HF_BUCKET = "baimamboukar/coordinate-concept-mismatch"
ARTIFACT_DATASET_KEY = "wildguardmix-v1"
ACTIVATION_ROWS_ENV = "ACTIVATION_ROWS_DIR"
ACTIVATION_STAGING_ENV = "ACTIVATION_STAGING_DIR"
BASELINE_ARTIFACT_ENV = "BASELINE_ARTIFACT_DIR"
EXPERIMENT_OUTPUT_ENV = "EXPERIMENT_OUTPUT_DIR"
HF_TOKEN_ENVIRONMENTS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")

CONFIG_SUFFIX = ".yaml"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
SEMANTIC_NAME_PATTERN = r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$"
GENERIC_EXPERIMENT_PATTERN = r"^(?:exp|experiment)_\d+$"
HF_COMMIT_PATTERN = r"^[0-9a-f]{40}$"
PIPELINE_STAGES = ("prepare", "preflight", "extract", "transfer", "align", "symmetry")

REQUIRED_BINARY_METRICS = frozenset(
    {
        "accuracy",
        "achieved_fpr_at_source_threshold",
        "auprc",
        "balanced_accuracy",
        "expected_calibration_error",
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
