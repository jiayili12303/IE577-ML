import numpy as np
import pytest

from jiayi_ml.supervised import GaussianNaiveBayes


def test_gaussian_naive_bayes_fits_simple_binary_data():
    X = np.array(
        [
            [-2.0, -2.0],
            [-1.5, -1.0],
            [-1.0, -1.5],
            [1.0, 1.5],
            [1.5, 1.0],
            [2.0, 2.0],
        ]
    )
    y = np.array([0, 0, 0, 1, 1, 1])

    model = GaussianNaiveBayes()
    model.fit(X, y)

    y_pred = model.predict(X)

    np.testing.assert_array_equal(y_pred, y)
    assert model.score(X, y) == pytest.approx(1.0)


def test_gaussian_naive_bayes_predict_proba_shape_and_sum():
    X = np.array(
        [
            [-2.0],
            [-1.0],
            [1.0],
            [2.0],
        ]
    )
    y = np.array([0, 0, 1, 1])

    model = GaussianNaiveBayes()
    model.fit(X, y)

    probabilities = model.predict_proba(X)

    assert probabilities.shape == (4, 2)
    assert np.all(probabilities >= 0.0)
    assert np.all(probabilities <= 1.0)
    np.testing.assert_allclose(np.sum(probabilities, axis=1), np.ones(4))


def test_gaussian_naive_bayes_predict_log_proba_shape():
    X = np.array(
        [
            [-2.0],
            [-1.0],
            [1.0],
            [2.0],
        ]
    )
    y = np.array(["negative", "negative", "positive", "positive"])

    model = GaussianNaiveBayes()
    model.fit(X, y)

    log_probabilities = model.predict_log_proba(X)

    assert log_probabilities.shape == (4, 2)


def test_gaussian_naive_bayes_supports_multiclass():
    X = np.array(
        [
            [0.0],
            [0.2],
            [5.0],
            [5.2],
            [10.0],
            [10.2],
        ]
    )
    y = np.array(["A", "A", "B", "B", "C", "C"])

    model = GaussianNaiveBayes()
    model.fit(X, y)

    predictions = model.predict(np.array([[0.1], [5.1], [10.1]]))

    np.testing.assert_array_equal(predictions, np.array(["A", "B", "C"]))


def test_gaussian_naive_bayes_predict_before_fit_raises_error():
    model = GaussianNaiveBayes()

    with pytest.raises(ValueError):
        model.predict(np.array([[1.0]]))


def test_gaussian_naive_bayes_rejects_shape_mismatch():
    X = np.array([[0.0], [1.0]])
    y = np.array([0, 1, 1])

    model = GaussianNaiveBayes()

    with pytest.raises(ValueError):
        model.fit(X, y)


def test_gaussian_naive_bayes_rejects_negative_var_smoothing():
    with pytest.raises(ValueError):
        GaussianNaiveBayes(var_smoothing=-1e-9)


def test_gaussian_naive_bayes_rejects_wrong_number_of_features():
    X_train = np.array([[0.0, 1.0], [1.0, 2.0]])
    y_train = np.array([0, 1])
    X_test = np.array([[1.0, 2.0, 3.0]])

    model = GaussianNaiveBayes()
    model.fit(X_train, y_train)

    with pytest.raises(ValueError):
        model.predict(X_test)