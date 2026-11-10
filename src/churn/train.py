from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.tree import DecisionTreeClassifier

from churn import config
from churn.preprocess import (
    Split,
    build_model_table,
    load_engineered,
    split_and_scale,
)


@dataclass(frozen=True)
class TrainedModel:
    name: str
    estimator: Any
    scaled: bool  # True if the model expects the scaled feature matrix

    def test_features(self, split: Split) -> pd.DataFrame:
        return split.X_test_scaled if self.scaled else split.X_test


def train_models(split: Split, seed: int = config.RANDOM_STATE) -> list[TrainedModel]:
    """Train the three baseline models used in the capstone."""
    log_reg = LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=seed
    )
    log_reg.fit(split.X_train_scaled, split.y_train)

    tree = DecisionTreeClassifier(
        max_depth=6, min_samples_leaf=10, class_weight="balanced", random_state=seed
    )
    tree.fit(split.X_train, split.y_train)

    forest = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    forest.fit(split.X_train, split.y_train)

    return [
        TrainedModel("Logistic Regression", log_reg, scaled=True),
        TrainedModel("Decision Tree", tree, scaled=False),
        TrainedModel("Random Forest", forest, scaled=False),
    ]


def compare_models(models: list[TrainedModel], split: Split) -> pd.DataFrame:
    """Test-set metrics for each model (decision threshold 0.5)."""
    rows = []
    for model in models:
        X_test = model.test_features(split)
        preds = model.estimator.predict(X_test)
        proba = model.estimator.predict_proba(X_test)[:, 1]
        rows.append(
            {
                "Model": model.name,
                "Precision": precision_score(split.y_test, preds, zero_division=0),
                "Recall": recall_score(split.y_test, preds, zero_division=0),
                "F1": f1_score(split.y_test, preds, zero_division=0),
                "ROC-AUC": roc_auc_score(split.y_test, proba),
                "PR-AUC": average_precision_score(split.y_test, proba),
            }
        )
    return pd.DataFrame(rows).round(3)


def save_models(
    models: list[TrainedModel], directory: Path = config.MODELS_DIR
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for model in models:
        filename = model.name.lower().replace(" ", "_") + ".pkl"
        joblib.dump(model.estimator, directory / filename)


def main() -> None:
    X, y = build_model_table(load_engineered())
    split = split_and_scale(X, y)
    models = train_models(split)
    print(compare_models(models, split).to_string(index=False))
    save_models(models)


if __name__ == "__main__":
    main()
