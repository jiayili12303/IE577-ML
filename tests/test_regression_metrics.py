import numpy as np
import pytest

from jiayi_ml.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)


def test_mean_squared_error():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 5.0])

    assert mean_squared_error(y_true, y_pred) == pytest.approx(4.0 / 3.0)


def test_root_mean_squared_error():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 5.0])

    assert root_mean_squared_error(y_true, y_pred) == pytest.approx(np.sqrt(4.0 / 3.0))


def test_mean_absolute_error():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 5.0])

    assert mean_absolute_error(y_true, y_pred) == pytest.approx(2.0 / 3.0)


def test_r2_score_perfect_prediction():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])

    assert r2_score(y_true, y_pred) == pytest.approx(1.0)


def test_r2_score_non_perfect_prediction():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 5.0])

    expected = 1 - 4.0 / 2.0
    assert r2_score(y_true, y_pred) == pytest.approx(expected)


def test_regression_metrics_reject_shape_mismatch():
    y_true = np.array([1.0, 2.0])
    y_pred = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError):
        mean_squared_error(y_true, y_pred)


def test_regression_metrics_reject_empty_input():
    with pytest.raises(ValueError):
        mean_squared_error([], [])