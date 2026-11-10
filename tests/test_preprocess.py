import numpy as np
import pandas as pd
import pytest

from churn import config
from churn.preprocess import build_model_table, split_and_scale


def test_build_model_table_drops_leakage_ids_and_target(
    accounts: pd.DataFrame,
) -> None:
    X, y = build_model_table(accounts)
    dropped = (
        config.LEAKAGE_COLS
        + config.REDUNDANT_COLS
        + config.ID_COLS
        + config.DATE_COLS
        + [config.TARGET]
    )
    assert not set(dropped) & set(X.columns)
    assert len(X) == len(y) == len(accounts)


def test_build_model_table_is_fully_numeric_without_missing(
    accounts: pd.DataFrame,
) -> None:
    X, _ = build_model_table(accounts)
    assert all(np.issubdtype(t, np.number) for t in X.dtypes)
    assert not X.isna().any().any()


def test_one_hot_encoding_drops_first_category(accounts: pd.DataFrame) -> None:
    X, _ = build_model_table(accounts)
    plan_cols = [c for c in X.columns if c.startswith("plan_tier_")]
    assert len(plan_cols) == 2  # 3 tiers, first one dropped


def test_missing_target_raises(accounts: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="target"):
        build_model_table(accounts.drop(columns=[config.TARGET]))


def test_missing_values_raise(accounts: pd.DataFrame) -> None:
    accounts.loc[0, "seats"] = np.nan
    with pytest.raises(ValueError, match="missing"):
        build_model_table(accounts)


def test_split_is_stratified(accounts: pd.DataFrame) -> None:
    X, y = build_model_table(accounts)
    split = split_and_scale(X, y)
    overall = y.mean()
    assert split.y_train.mean() == pytest.approx(overall, abs=0.02)
    assert split.y_test.mean() == pytest.approx(overall, abs=0.02)


def test_scaler_is_fit_on_train_only(accounts: pd.DataFrame) -> None:
    X, y = build_model_table(accounts)
    split = split_and_scale(X, y)
    assert split.X_train_scaled["seats"].mean() == pytest.approx(0, abs=1e-9)
    # If the scaler had seen the test set, its mean would also be exactly 0
    assert abs(split.X_test_scaled["seats"].mean()) > 1e-6


def test_binary_columns_are_not_scaled(accounts: pd.DataFrame) -> None:
    X, y = build_model_table(accounts)
    split = split_and_scale(X, y)
    pd.testing.assert_series_equal(
        split.X_train_scaled["is_trial"], split.X_train["is_trial"]
    )


def test_split_is_reproducible(accounts: pd.DataFrame) -> None:
    X, y = build_model_table(accounts)
    first = split_and_scale(X, y, seed=7)
    second = split_and_scale(X, y, seed=7)
    pd.testing.assert_frame_equal(first.X_test, second.X_test)
