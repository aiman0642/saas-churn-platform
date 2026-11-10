from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
ENGINEERED_FILE = PROCESSED_DIR / "account_view_engineered.csv"

RANDOM_STATE = 42
TARGET = "churn_flag"

# Columns that reveal the label or duplicate other columns
LEAKAGE_COLS = ["churn_date"]
REDUNDANT_COLS = [
    "tenure_bucket",
    "usage_bucket",
    "mrr_bucket",
    "plan_tier_sub",
    "last_activity_date",
]
ID_COLS = ["account_id", "account_name"]
DATE_COLS = ["signup_date"]
CATEGORICAL_COLS = [
    "industry",
    "country",
    "plan_tier",
    "billing_frequency",
    "referral_source",
]
