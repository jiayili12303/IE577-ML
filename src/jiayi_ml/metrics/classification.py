"""
Classification evaluation metrics implemented from scratch using NumPy.
"""

from __future__ import annotations

import numpy as np


def _validate_classification_inputs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Validate classification target and prediction arrays.

    Parameters
    ----------
    y_true : array-like
        True class labels.
    y_pred : array-like
        Predicted class labels.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Validated one-dimensional NumPy arrays.

    Raises
    ------
    ValueError
        If the inputs have different shapes or are empty.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    if y_true.size == 0:
        raise ValueError("y_true and y_pred must not be empty.")

    return y_true, y_pred


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute classification accuracy.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True class labels.
    y_pred : array-like of shape (n_samples,)
        Predicted class labels.

    Returns
    -------
    float
        Fraction of correctly classified samples.
    """
    y_true, y_pred = _validate_classification_inputs(y_true, y_pred)
    return float(np.mean(y_true == y_pred))


def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list | np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute a confusion matrix.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True class labels.
    y_pred : array-like of shape (n_samples,)
        Predicted class labels.
    labels : list or np.ndarray, optional
        Ordered list of labels to index the matrix. If None, labels are inferred
        from the union of y_true and y_pred and sorted.

    Returns
    -------
    np.ndarray
        Confusion matrix of shape (n_classes, n_classes), where rows correspond
        to true labels and columns correspond to predicted labels.
    """
    y_true, y_pred = _validate_classification_inputs(y_true, y_pred)

    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    else:
        labels = np.asarray(labels)

    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    label_to_index = {label: index for index, label in enumerate(labels)}

    for true_label, pred_label in zip(y_true, y_pred):
        if true_label not in label_to_index or pred_label not in label_to_index:
            raise ValueError("All labels in y_true and y_pred must appear in labels.")

        matrix[label_to_index[true_label], label_to_index[pred_label]] += 1

    return matrix


def precision_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pos_label: int | str = 1,
) -> float:
    """
    Compute binary classification precision.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True class labels.
    y_pred : array-like of shape (n_samples,)
        Predicted class labels.
    pos_label : int or str, default=1
        Label treated as the positive class.

    Returns
    -------
    float
        Precision score. If there are no predicted positives, returns 0.0.
    """
    y_true, y_pred = _validate_classification_inputs(y_true, y_pred)

    true_positive = np.sum((y_true == pos_label) & (y_pred == pos_label))
    false_positive = np.sum((y_true != pos_label) & (y_pred == pos_label))

    denominator = true_positive + false_positive
    if denominator == 0:
        return 0.0

    return float(true_positive / denominator)


def recall_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pos_label: int | str = 1,
) -> float:
    """
    Compute binary classification recall.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True class labels.
    y_pred : array-like of shape (n_samples,)
        Predicted class labels.
    pos_label : int or str, default=1
        Label treated as the positive class.

    Returns
    -------
    float
        Recall score. If there are no true positives and false negatives,
        returns 0.0.
    """
    y_true, y_pred = _validate_classification_inputs(y_true, y_pred)

    true_positive = np.sum((y_true == pos_label) & (y_pred == pos_label))
    false_negative = np.sum((y_true == pos_label) & (y_pred != pos_label))

    denominator = true_positive + false_negative
    if denominator == 0:
        return 0.0

    return float(true_positive / denominator)


def f1_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pos_label: int | str = 1,
) -> float:
    """
    Compute binary classification F1 score.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True class labels.
    y_pred : array-like of shape (n_samples,)
        Predicted class labels.
    pos_label : int or str, default=1
        Label treated as the positive class.

    Returns
    -------
    float
        Harmonic mean of precision and recall. If both precision and recall are
        zero, returns 0.0.
    """
    precision = precision_score(y_true, y_pred, pos_label=pos_label)
    recall = recall_score(y_true, y_pred, pos_label=pos_label)

    denominator = precision + recall
    if denominator == 0:
        return 0.0

    return float(2 * precision * recall / denominator)