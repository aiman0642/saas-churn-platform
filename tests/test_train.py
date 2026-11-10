import numpy as np
import pandas as pd
import pytest

from churn import config
from churn.preprocess import build_model_table, load_engineered, split_and_scale
from churn.train import compare_models, save_models, train_models


@pytest.fixture
def trained(accounts: pd.DataFrame):  # type: ignore[no-untyped-def]
    X, y = build_model_table(accounts)
    split = split_and_scale(X, y)
    return split, train_models(split)


def test_trains_three_named_models(trained) -> None:  # type: ignore[no-untyped-def]
    _, models = trained
    assert [m.name for m in models] == [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
    ]


def test_training_is_deterministic(trained) -> None:  # type: ignore[no-untyped-def]
    split, models = trained
    again = train_models(split)
    for first, second in zip(models, again, strict=True):
        X = first.test_features(split)
        assert np.array_equal(first.estimator.predict(X), second.estimator.predict(X))


def test_compare_models_returns_valid_metrics(trained) -> None:  # type: ignore[no-untyped-def]
    split, models = trained
    table = compare_models(models, split)
    assert list(table.columns) == [
        "Model",
        "Precision",
        "Recall",
        "F1",
        "ROC-AUC",
        "PR-AUC",
    ]
    metrics = table.drop(columns="Model")
    assert ((metrics >= 0) & (metrics <= 1)).all().all()


def test_models_beat_random_on_signal_data(trained) -> None:  # type: ignore[no-untyped-def]
    split, models = trained
    table = compare_models(models, split).set_index("Model")
    assert table.loc["Logistic Regression", "ROC-AUC"] > 0.6


def test_save_models_writes_files(trained, tmp_path) -> None:  # type: ignore[no-untyped-def]
    _, models = trained
    save_models(models, tmp_path)
    assert (tmp_path / "logistic_regression.pkl").exists()
    assert (tmp_path / "random_forest.pkl").exists()


@pytest.mark.skipif(
    not config.ENGINEERED_FILE.exists(), reason="real data not available (e.g. CI)"
)
def test_logistic_regression_reproduces_capstone_result() -> None:
    X, y = build_model_table(load_engineered())
    split = split_and_scale(X, y)
    table = compare_models(train_models(split), split).set_index("Model")
    assert table.loc["Logistic Regression", "F1"] == pytest.approx(0.471, abs=0.005)
    assert table.loc["Logistic Regression", "ROC-AUC"] == pytest.approx(
        0.684, abs=0.005
    )
