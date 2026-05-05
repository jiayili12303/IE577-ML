import numpy as np
import pytest

from jiayi_ml.supervised import LinearSVM


def test_linear_svm_fits_linearly_separable_data():
    X = np.array(
        [
            [-2.0, -2.0],
            [-1.0, -1.0],
            [1.0, 1.0],
            [2.0, 2.0],
        ]
    )
    y = np.array([0, 0, 1, 1])

    model = LinearSVM(learning_rate=0.05, max_iter=5000, alpha=0.01)
    model.fit(X, y)

    predictions = model.predict(X)

    np.testing.assert_array_equal(predictions, y)
    assert model.score(X, y) == pytest.approx(1.0)


def test_linear_svm_supports_string_labels():
    X = np.array(
        [
            [-2.0],
            [-1.0],
            [1.0],
            [2.0],
        ]
    )
    y = np.array(["negative", "negative", "positive", "positive"])

    model = LinearSVM(learning_rate=0.05, max_iter=5000, alpha=0.01)
    model.fit(X, y)

    predictions = model.predict(np.array([[-1.5], [1.5]]))

    np.testing.assert_array_equal(
        predictions,
        np.array(["negative", "positive"]),
    )


def test_linear_svm_decision_function_shape():
    X = np.array(
        [
            [-2.0],
            [-1.0],
            [1.0],
            [2.0],
        ]
    )
    y = np.array([0, 0, 1, 1])

    model = LinearSVM(learning_rate=0.05, max_iter=5000, alpha=0.01)
    model.fit(X, y)

    scores = model.decision_function(X)

    assert scores.shape == (4,)


def test_linear_svm_predict_before_fit_raises_error():
    model = LinearSVM()

    with pytest.raises(ValueError):
        model.predict(np.array([[1.0]]))


def test_linear_svm_rejects_multiclass_target():
    X = np.array(
        [
            [0.0],
            [1.0],
            [2.0],
        ]
    )
    y = np.array([0, 1, 2])

    model = LinearSVM()

    with pytest.raises(ValueError):
        model.fit(X, y)


def test_linear_svm_rejects_invalid_learning_rate():
    with pytest.raises(ValueError):
        LinearSVM(learning_rate=0.0)


def test_linear_svm_rejects_invalid_max_iter():
    with pytest.raises(ValueError):
        LinearSVM(max_iter=0)


def test_linear_svm_rejects_negative_alpha():
    with pytest.raises(ValueError):
        LinearSVM(alpha=-0.1)


def test_linear_svm_rejects_negative_tol():
    with pytest.raises(ValueError):
        LinearSVM(tol=-1e-3)


def test_linear_svm_rejects_shape_mismatch():
    X = np.array([[0.0], [1.0]])
    y = np.array([0, 1, 1])

    model = LinearSVM()

    with pytest.raises(ValueError):
        model.fit(X, y)


def test_linear_svm_rejects_wrong_number_of_features():
    X = np.array(
        [
            [-2.0, -2.0],
            [-1.0, -1.0],
            [1.0, 1.0],
            [2.0, 2.0],
        ]
    )
    y = np.array([0, 0, 1, 1])

    model = LinearSVM(learning_rate=0.05, max_iter=5000, alpha=0.01)
    model.fit(X, y)

    with pytest.raises(ValueError):
        model.predict(np.array([[1.0, 2.0, 3.0]]))


def test_linear_svm_without_intercept_runs():
    X = np.array(
        [
            [-2.0],
            [-1.0],
            [1.0],
            [2.0],
        ]
    )
    y = np.array([0, 0, 1, 1])

    model = LinearSVM(
        learning_rate=0.05,
        max_iter=5000,
        alpha=0.01,
        fit_intercept=False,
    )
    model.fit(X, y)

    predictions = model.predict(X)

    np.testing.assert_array_equal(predictions, y)
    assert model.intercept_ == pytest.approx(0.0)