"""
Regression evaluation metrics implemented from scratch using NumPy.
"""

from __future__ import annotations

import numpy as np


def _validate_regression_inputs(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Validate regression target and prediction arrays.

    Parameters
    ----------
    y_true : array-like
        True target values.
    y_pred : array-like
        Predicted target values.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Validated one-dimensional NumPy arrays.

    Raises
    ------
    ValueError
        If the inputs have different shapes or are empty.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    if y_true.size == 0:
        raise ValueError("y_true and y_pred must not be empty.")

    return y_true, y_pred


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the mean squared error.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True target values.
    y_pred : array-like of shape (n_samples,)
        Predicted target values.

    Returns
    -------
    float
        Mean squared error.
    """
    y_true, y_pred = _validate_regression_inputs(y_true, y_pred)
    return float(np.mean((y_true - y_pred) ** 2))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the root mean squared error.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True target values.
    y_pred : array-like of shape (n_samples,)
        Predicted target values.

    Returns
    -------
    float
        Root mean squared error.
    """
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the mean absolute error.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True target values.
    y_pred : array-like of shape (n_samples,)
        Predicted target values.

    Returns
    -------
    float
        Mean absolute error.
    """
    y_true, y_pred = _validate_regression_inputs(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the coefficient of determination, R-squared.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True target values.
    y_pred : array-like of shape (n_samples,)
        Predicted target values.

    Returns
    -------
    float
        R-squared score.

    Notes
    -----
    R-squared is defined as:

        1 - SS_res / SS_tot

    If all true target values are constant, this function returns 0.0.
    """
    y_true, y_pred = _validate_regression_inputs(y_true, y_pred)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0:
        return 0.0

    return float(1 - ss_res / ss_tot)