import numpy as np
import pytest

from jiayi_ml.supervised import KNNClassifier, KNNRegressor


def test_knn_classifier_predicts_simple_classes():
    X = np.array(
        [
            [0.0],
            [1.0],
            [10.0],
            [11.0],
        ]
    )
    y = np.array(["A", "A", "B", "B"])

    model = KNNClassifier(n_neighbors=1)
    model.fit(X, y)

    predictions = model.predict(np.array([[0.2], [10.5]]))

    np.testing.assert_array_equal(predictions, np.array(["A", "B"]))


def test_knn_classifier_score():
    X = np.array(
        [
            [0.0],
            [1.0],
            [10.0],
            [11.0],
        ]
    )
    y = np.array([0, 0, 1, 1])

    model = KNNClassifier(n_neighbors=1)
    model.fit(X, y)

    assert model.score(X, y) == pytest.approx(1.0)


def test_knn_classifier_distance_weights():
    X = np.array(
        [
            [0.0],
            [2.0],
            [10.0],
        ]
    )
    y = np.array(["A", "B", "B"])

    model = KNNClassifier(n_neighbors=3, weights="distance")
    model.fit(X, y)

    prediction = model.predict(np.array([[0.1]]))

    np.testing.assert_array_equal(prediction, np.array(["A"]))


def test_knn_classifier_manhattan_metric():
    X = np.array(
        [
            [0.0, 0.0],
            [5.0, 5.0],
        ]
    )
    y = np.array([0, 1])

    model = KNNClassifier(n_neighbors=1, metric="manhattan")
    model.fit(X, y)

    prediction = model.predict(np.array([[4.0, 4.0]]))

    np.testing.assert_array_equal(prediction, np.array([1]))


def test_knn_classifier_predict_before_fit_raises_error():
    model = KNNClassifier(n_neighbors=1)

    with pytest.raises(ValueError):
        model.predict(np.array([[1.0]]))


def test_knn_classifier_rejects_invalid_k():
    with pytest.raises(ValueError):
        KNNClassifier(n_neighbors=0)


def test_knn_classifier_rejects_too_many_neighbors():
    X = np.array([[0.0], [1.0]])
    y = np.array([0, 1])

    model = KNNClassifier(n_neighbors=3)

    with pytest.raises(ValueError):
        model.fit(X, y)


def test_knn_classifier_rejects_wrong_number_of_features():
    X = np.array([[0.0, 1.0], [1.0, 2.0]])
    y = np.array([0, 1])

    model = KNNClassifier(n_neighbors=1)
    model.fit(X, y)

    with pytest.raises(ValueError):
        model.predict(np.array([[1.0, 2.0, 3.0]]))


def test_knn_regressor_predicts_average_of_neighbors():
    X = np.array(
        [
            [0.0],
            [1.0],
            [10.0],
            [11.0],
        ]
    )
    y = np.array([0.0, 2.0, 10.0, 12.0])

    model = KNNRegressor(n_neighbors=2)
    model.fit(X, y)

    prediction = model.predict(np.array([[0.5]]))

    np.testing.assert_allclose(prediction, np.array([1.0]))


def test_knn_regressor_distance_weights():
    X = np.array(
        [
            [0.0],
            [10.0],
        ]
    )
    y = np.array([0.0, 10.0])

    model = KNNRegressor(n_neighbors=2, weights="distance")
    model.fit(X, y)

    prediction = model.predict(np.array([[1.0]]))

    assert prediction[0] < 5.0


def test_knn_regressor_exact_match_with_distance_weights():
    X = np.array(
        [
            [0.0],
            [10.0],
        ]
    )
    y = np.array([3.0, 10.0])

    model = KNNRegressor(n_neighbors=2, weights="distance")
    model.fit(X, y)

    prediction = model.predict(np.array([[0.0]]))

    np.testing.assert_allclose(prediction, np.array([3.0]))


def test_knn_regressor_score_perfect_on_training_with_k_one():
    X = np.array(
        [
            [0.0],
            [1.0],
            [2.0],
        ]
    )
    y = np.array([1.0, 2.0, 3.0])

    model = KNNRegressor(n_neighbors=1)
    model.fit(X, y)

    assert model.score(X, y) == pytest.approx(1.0)


def test_knn_regressor_predict_before_fit_raises_error():
    model = KNNRegressor(n_neighbors=1)

    with pytest.raises(ValueError):
        model.predict(np.array([[1.0]]))


def test_knn_regressor_rejects_invalid_weights():
    with pytest.raises(ValueError):
        KNNRegressor(weights="bad_weights")