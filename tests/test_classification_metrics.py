import numpy as np
import pytest

from jiayi_ml.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def test_accuracy_score():
    y_true = np.array([1, 0, 1, 1])
    y_pred = np.array([1, 0, 0, 1])

    assert accuracy_score(y_true, y_pred) == pytest.approx(0.75)


def test_confusion_matrix_binary():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])

    expected = np.array(
        [
            [1, 1],
            [1, 1],
        ]
    )

    np.testing.assert_array_equal(confusion_matrix(y_true, y_pred), expected)


def test_confusion_matrix_with_custom_labels():
    y_true = np.array(["cat", "cat", "dog", "dog"])
    y_pred = np.array(["cat", "dog", "cat", "dog"])

    expected = np.array(
        [
            [1, 1],
            [1, 1],
        ]
    )

    np.testing.assert_array_equal(
        confusion_matrix(y_true, y_pred, labels=["cat", "dog"]),
        expected,
    )


def test_precision_score():
    y_true = np.array([1, 0, 1, 1])
    y_pred = np.array([1, 1, 0, 1])

    assert precision_score(y_true, y_pred) == pytest.approx(2.0 / 3.0)


def test_recall_score():
    y_true = np.array([1, 0, 1, 1])
    y_pred = np.array([1, 1, 0, 1])

    assert recall_score(y_true, y_pred) == pytest.approx(2.0 / 3.0)


def test_f1_score():
    y_true = np.array([1, 0, 1, 1])
    y_pred = np.array([1, 1, 0, 1])

    assert f1_score(y_true, y_pred) == pytest.approx(2.0 / 3.0)


def test_precision_score_no_predicted_positives():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([0, 0, 0, 0])

    assert precision_score(y_true, y_pred) == 0.0


def test_recall_score_no_true_positives():
    y_true = np.array([0, 0, 0, 0])
    y_pred = np.array([1, 0, 1, 0])

    assert recall_score(y_true, y_pred) == 0.0


def test_classification_metrics_reject_shape_mismatch():
    y_true = np.array([1, 0])
    y_pred = np.array([1, 0, 1])

    with pytest.raises(ValueError):
        accuracy_score(y_true, y_pred)


def test_classification_metrics_reject_empty_input():
    with pytest.raises(ValueError):
        accuracy_score([], [])