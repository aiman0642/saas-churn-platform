import pytest

from mltemplate.utils import group_train_test_split, precision_recall_f1


def test_split_has_no_group_leakage() -> None:
    groups = ["a", "a", "b", "b", "c", "d", "e", "e"]
    train, test = group_train_test_split(groups, test_size=0.4)
    assert train.isdisjoint(test)
    assert train | test == set(groups)


def test_split_is_deterministic_for_same_seed() -> None:
    groups = list("abcdefghij")
    assert group_train_test_split(groups, seed=1) == group_train_test_split(
        groups, seed=1
    )


def test_split_rejects_bad_test_size() -> None:
    with pytest.raises(ValueError):
        group_train_test_split(["a", "b"], test_size=1.5)


def test_metrics_perfect_prediction() -> None:
    result = precision_recall_f1([1, 0, 1, 0], [1, 0, 1, 0])
    assert result == {"precision": 1.0, "recall": 1.0, "f1": 1.0}


def test_metrics_known_values() -> None:
    result = precision_recall_f1([1, 1, 0, 0], [1, 0, 1, 0])
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5
    assert result["f1"] == 0.5


def test_metrics_no_positives_returns_zeros() -> None:
    result = precision_recall_f1([0, 0], [0, 0])
    assert result == {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def test_metrics_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        precision_recall_f1([1, 0], [1])


@pytest.fixture
def subjects() -> list[str]:
    return [f"person_{i}" for i in range(20)]


@pytest.mark.parametrize(
    ("test_size", "expected_test_groups"),
    [(0.1, 2), (0.25, 5), (0.5, 10)],
)
def test_split_sizes(
    subjects: list[str], test_size: float, expected_test_groups: int
) -> None:
    train, test = group_train_test_split(subjects, test_size=test_size)
    assert len(test) == expected_test_groups
    assert len(train) == 20 - expected_test_groups


@pytest.mark.parametrize(
    ("y_true", "y_pred", "expected_f1"),
    [
        ([1, 1, 1, 1], [1, 1, 1, 1], 1.0),
        ([1, 1, 0, 0], [0, 0, 1, 1], 0.0),
        ([1, 0, 0, 0], [1, 1, 0, 0], pytest.approx(2 / 3)),
    ],
)
def test_f1_cases(y_true: list[int], y_pred: list[int], expected_f1: float) -> None:
    assert precision_recall_f1(y_true, y_pred)["f1"] == expected_f1
