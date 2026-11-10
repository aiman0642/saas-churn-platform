from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from churn import config


@dataclass(frozen=True)
class Split:
    """Train/test data in raw and scaled form, plus the fitted scaler."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    X_train_scaled: pd.DataFrame
    X_test_scaled: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    scaler: StandardScaler


def load_engineered(path: Path = config.ENGINEERED_FILE) -> pd.DataFrame:
    """Load the account table produced by the feature-engineering step."""
    return pd.read_csv(path, parse_dates=["signup_date", "churn_date"])


def build_model_table(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Drop leakage/ID/redundant columns, one-hot encode, return (X, y)."""
    if config.TARGET not in df.columns:
        raise ValueError(f"missing target column: {config.TARGET}")

    to_drop = (
        config.LEAKAGE_COLS + config.REDUNDANT_COLS + config.ID_COLS + config.DATE_COLS
    )
    table = df.drop(columns=[c for c in to_drop if c in df.columns])

    categorical = [c for c in config.CATEGORICAL_COLS if c in table.columns]
    table = pd.get_dummies(table, columns=categorical, drop_first=True)

    bool_cols = table.select_dtypes(include="bool").columns.tolist()
    table[bool_cols] = table[bool_cols].astype(int)

    y = table[config.TARGET]
    X = table.drop(columns=[config.TARGET])

    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        raise ValueError(f"non-numeric columns remain: {non_numeric}")
    if X.isna().any().any():
        raise ValueError("missing values remain in the feature table")
    return X, y


def split_and_scale(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    seed: int = config.RANDOM_STATE,
) -> Split:
    """Stratified split; the scaler is fit on the training set only."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    # Scale continuous columns only, not 0/1 flags or one-hot columns
    numeric = [c for c in X_train.columns if X_train[c].nunique() > 2]

    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[numeric] = scaler.fit_transform(X_train[numeric])
    X_test_scaled[numeric] = scaler.transform(X_test[numeric])

    return Split(
        X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler
    )
