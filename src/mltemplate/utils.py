from __future__ import annotations

import random
from collections.abc import Sequence
from typing import TypeVar

T = TypeVar("T")


def group_train_test_split(
    groups: Sequence[T], test_size: float = 0.2, seed: int = 42
) -> tuple[set[T], set[T]]:
    """Split unique group ids into train/test sets so no group is in both.

    Use this when one subject (person, patient, user) has many rows, to avoid
    data leakage across splits.
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    unique = sorted(set(groups), key=str)
    if len(unique) < 2:
        raise ValueError("need at least 2 unique groups")
    rng = random.Random(seed)
    rng.shuffle(unique)
    n_test = min(max(1, round(len(unique) * test_size)), len(unique) - 1)
    return set(unique[n_test:]), set(unique[:n_test])


def precision_recall_f1(
    y_true: Sequence[int], y_pred: Sequence[int]
) -> dict[str, float]:
    """Binary precision, recall and F1 (positive class = 1)."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}
