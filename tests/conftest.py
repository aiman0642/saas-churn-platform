import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def accounts() -> pd.DataFrame:
    """Small synthetic table with the same columns as the engineered data."""
    rng = np.random.default_rng(0)
    n = 300
    churn = rng.random(n) < 0.25
    return pd.DataFrame(
        {
            "account_id": [f"A-{i}" for i in range(n)],
            "account_name": [f"Company {i}" for i in range(n)],
            "industry": rng.choice(["SaaS", "Retail", "Health"], n),
            "country": rng.choice(["PK", "US", "UK"], n),
            "signup_date": pd.Timestamp("2023-01-01"),
            "referral_source": rng.choice(["ads", "organic"], n),
            "plan_tier": rng.choice(["basic", "pro", "enterprise"], n),
            "plan_tier_sub": "dup",
            "billing_frequency": rng.choice(["monthly", "annual"], n),
            "tenure_bucket": "x",
            "usage_bucket": "x",
            "mrr_bucket": "x",
            "last_activity_date": "2024-01-01",
            "churn_date": pd.NaT,
            "is_trial": rng.random(n) < 0.1,
            "churn_flag": churn,
            "seats": rng.integers(1, 100, n),
            "mrr_amount": rng.integers(50, 5000, n),
            "total_usage_count": rng.integers(0, 10_000, n) - churn * 500,
            "engagement_trend_ratio": rng.normal(1.0, 0.3, n) - churn * 0.3,
        }
    )
