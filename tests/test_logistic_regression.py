import numpy as np
import pytest

from jiayi_ml.supervised import LogisticRegression


def test_logistic_regression_fits_simple_binary_data():
    X = np.array(
        [
            [-2.0],
            [-1.0],
            [0.0],
            [1.0],
            [2.0],
            [3.0],
        ]
    )
    y = np.array([0, 0, 0, 1, 1, 1])

    model = LogisticRegression(learning_rate=0.2, max_iter=5000, tol=1e-10)
    model.fit(X, y)

    y_pred = model.predict(X)

    np.testing.assert_array_equal(y_pred, y)
    assert model.score(X, y) == pytest.approx(1.0)


def test_logistic_regression_predict_proba_shape_and_range():
    X = np.array(
        [
            [-2.0],
            [-1.0],
            [1.0],
            [2.0],
        ]
    )
    y = np.array([0, 0, 1, 1])

    model = LogisticRegression(learning_rate=0.2, max_iter=2000)
    model.fit(X, y)

    probabilities = model.predict_proba(X)

    assert probabilities.shape == (4, 2)
    assert np.all(probabilities >= 0.0)
    assert np.all(probabilities <= 1.0)
    np.testing.assert_allclose(np.sum(probabilities, axis=1), np.ones(4))


def test_logistic_regression_supports_non_numeric_labels():
    X = np.array(
        [
            [-2.0],
            [-1.0],
            [1.0],
            [2.0],
        ]
    )
    y = np.array(["negative", "negative", "positive", "positive"])

    model = LogisticRegression(learning_rate=0.2, max_iter=3000)
    model.fit(X, y)

    y_pred = model.predict(X)

    np.testing.assert_array_equal(y_pred, y)


def test_logistic_regression_predict_before_fit_raises_error():
    model = LogisticRegression()

    with pytest.raises(ValueError):
        model.predict(np.array([[1.0]]))


def test_logistic_regression_rejects_multiclass_target():
    X = np.array(
        [
            [0.0],
            [1.0],
            [2.0],
        ]
    )
    y = np.array([0, 1, 2])

    model = LogisticRegression()

    with pytest.raises(ValueError):
        model.fit(X, y)


def test_logistic_regression_rejects_shape_mismatch():
    X = np.array([[0.0], [1.0]])
    y = np.array([0, 1, 1])

    model = LogisticRegression()

    with pytest.raises(ValueError):
        model.fit(X, y)


def test_logistic_regression_rejects_invalid_threshold():
    X = np.array(
        [
            [-1.0],
            [1.0],
        ]
    )
    y = np.array([0, 1])

    model = LogisticRegression(learning_rate=0.2, max_iter=1000)
    model.fit(X, y)

    with pytest.raises(ValueError):
        model.predict(X, threshold=1.5)


def test_logistic_regression_rejects_invalid_penalty():
    with pytest.raises(ValueError):
        LogisticRegression(penalty="elasticnet")


def test_logistic_regression_l2_regularization_runs():
    X = np.array(
        [
            [-2.0, 0.0],
            [-1.0, 0.5],
            [1.0, 1.0],
            [2.0, 1.5],
        ]
    )
    y = np.array([0, 0, 1, 1])

    model = LogisticRegression(
        learning_rate=0.1,
        max_iter=3000,
        penalty="l2",
        alpha=0.01,
    )
    model.fit(X, y)

    y_pred = model.predict(X)

    assert y_pred.shape == y.shape