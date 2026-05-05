import numpy as np
import pytest

from jiayi_ml.supervised import Perceptron


def test_perceptron_fits_linearly_separable_data():
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )
    y = np.array([0, 0, 0, 1])

    model = Perceptron(learning_rate=0.1, max_iter=100, random_state=438)
    model.fit(X, y)

    predictions = model.predict(X)

    np.testing.assert_array_equal(predictions, y)
    assert model.score(X, y) == pytest.approx(1.0)


def test_perceptron_supports_string_labels():
    X = np.array(
        [
            [0.0],
            [1.0],
            [2.0],
            [3.0],
        ]
    )
    y = np.array(["low", "low", "high", "high"])

    model = Perceptron(learning_rate=0.1, max_iter=100, shuffle=False)
    model.fit(X, y)

    predictions = model.predict(np.array([[0.5], [2.5]]))

    np.testing.assert_array_equal(predictions, np.array(["low", "high"]))


def test_perceptron_decision_function_shape():
    X = np.array(
        [
            [0.0],
            [1.0],
            [2.0],
            [3.0],
        ]
    )
    y = np.array([0, 0, 1, 1])

    model = Perceptron(learning_rate=0.1, max_iter=100)
    model.fit(X, y)

    scores = model.decision_function(X)

    assert scores.shape == (4,)


def test_perceptron_without_intercept_runs():
    X = np.array(
        [
            [-2.0],
            [-1.0],
            [1.0],
            [2.0],
        ]
    )
    y = np.array([0, 0, 1, 1])

    model = Perceptron(
        learning_rate=0.1,
        max_iter=100,
        fit_intercept=False,
        shuffle=False,
    )
    model.fit(X, y)

    predictions = model.predict(X)

    np.testing.assert_array_equal(predictions, y)
    assert model.intercept_ == pytest.approx(0.0)


def test_perceptron_predict_before_fit_raises_error():
    model = Perceptron()

    with pytest.raises(ValueError):
        model.predict(np.array([[1.0]]))


def test_perceptron_rejects_multiclass_target():
    X = np.array(
        [
            [0.0],
            [1.0],
            [2.0],
        ]
    )
    y = np.array([0, 1, 2])

    model = Perceptron()

    with pytest.raises(ValueError):
        model.fit(X, y)


def test_perceptron_rejects_invalid_learning_rate():
    with pytest.raises(ValueError):
        Perceptron(learning_rate=0.0)


def test_perceptron_rejects_invalid_max_iter():
    with pytest.raises(ValueError):
        Perceptron(max_iter=0)


def test_perceptron_rejects_shape_mismatch():
    X = np.array([[0.0], [1.0]])
    y = np.array([0, 1, 1])

    model = Perceptron()

    with pytest.raises(ValueError):
        model.fit(X, y)


def test_perceptron_rejects_wrong_number_of_features():
    X = np.array(
        [
            [0.0, 1.0],
            [1.0, 2.0],
            [2.0, 3.0],
        ]
    )
    y = np.array([0, 0, 1])

    model = Perceptron()
    model.fit(X, y)

    with pytest.raises(ValueError):
        model.predict(np.array([[1.0, 2.0, 3.0]]))