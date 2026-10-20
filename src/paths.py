"""Project-root-relative filesystem locations shared by every module."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RAW_CSV = ROOT / "data" / "raw" / "creditcard.csv"
PROCESSED_DIR = ROOT / "data" / "processed"

MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "xgb_model.joblib"
FEATURE_NAMES_PATH = MODELS_DIR / "feature_names.json"

RESULTS_DIR = ROOT / "results"
EVAL_REPORT_PATH = RESULTS_DIR / "evaluation_report.json"

LOGS_DIR = ROOT / "logs"
EXPLANATION_LOG_PATH = LOGS_DIR / "explanation_log.jsonl"
DRIFT_LOG_PATH = LOGS_DIR / "drift_log.json"

TARGET = "Class"
SEED = 42
