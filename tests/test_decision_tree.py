import numpy as np
import pytest

from jiayi_ml.supervised import DecisionTreeClassifier, DecisionTreeRegressor


def test_decision_tree_classifier_fits_simple_binary_data():
    X = np.array(
        [
            [0.0],
            [1.0],
            [2.0],
            [3.0],
        ]
    )
    y = np.array([0, 0, 1, 1])

    model = DecisionTreeClassifier(max_depth=2)
    model.fit(X, y)

    predictions = model.predict(X)

    np.testing.assert_array_equal(predictions, y)
    assert model.score(X, y) == pytest.approx(1.0)


def test_decision_tree_classifier_supports_string_labels():
    X = np.array(
        [
            [0.0],
            [1.0],
            [2.0],
            [3.0],
        ]
    )
    y = np.array(["low", "low", "high", "high"])

    model = DecisionTreeClassifier(max_depth=2)
    model.fit(X, y)

    predictions = model.predict(np.array([[0.5], [2.5]]))

    np.testing.assert_array_equal(predictions, np.array(["low", "high"]))


def test_decision_tree_classifier_entropy_runs():
    X = np.array(
        [
            [0.0],
            [1.0],
            [2.0],
            [3.0],
        ]
    )
    y = np.array([0, 0, 1, 1])

    model = DecisionTreeClassifier(criterion="entropy")
    model.fit(X, y)

    assert model.predict(X).shape == y.shape


def test_decision_tree_classifier_feature_importances():
    X = np.array(
        [
            [0.0, 10.0],
            [1.0, 10.0],
            [2.0, 10.0],
            [3.0, 10.0],
        ]
    )
    y = np.array([0, 0, 1, 1])

    model = DecisionTreeClassifier(max_depth=1)
    model.fit(X, y)

    assert model.feature_importances_.shape == (2,)
    assert model.feature_importances_[0] > model.feature_importances_[1]
    assert np.sum(model.feature_importances_) == pytest.approx(1.0)


def test_decision_tree_classifier_predict_before_fit_raises_error():
    model = DecisionTreeClassifier()

    with pytest.raises(ValueError):
        model.predict(np.array([[1.0]]))


def test_decision_tree_classifier_rejects_invalid_criterion():
    with pytest.raises(ValueError):
        DecisionTreeClassifier(criterion="bad")


def test_decision_tree_classifier_rejects_wrong_number_of_features():
    X = np.array(
        [
            [0.0, 1.0],
            [1.0, 2.0],
            [2.0, 3.0],
        ]
    )
    y = np.array([0, 0, 1])

    model = DecisionTreeClassifier()
    model.fit(X, y)

    with pytest.raises(ValueError):
        model.predict(np.array([[1.0, 2.0, 3.0]]))


def test_decision_tree_regressor_fits_simple_step_function():
    X = np.array(
        [
            [0.0],
            [1.0],
            [2.0],
            [3.0],
        ]
    )
    y = np.array([1.0, 1.0, 5.0, 5.0])

    model = DecisionTreeRegressor(max_depth=2)
    model.fit(X, y)

    predictions = model.predict(X)

    np.testing.assert_allclose(predictions, y)
    assert model.score(X, y) == pytest.approx(1.0)


def test_decision_tree_regressor_prediction_shape():
    X = np.array(
        [
            [0.0],
            [1.0],
            [2.0],
            [3.0],
        ]
    )
    y = np.array([0.0, 1.0, 2.0, 3.0])

    model = DecisionTreeRegressor(max_depth=2)
    model.fit(X, y)

    predictions = model.predict(np.array([[0.5], [2.5]]))

    assert predictions.shape == (2,)


def test_decision_tree_regressor_absolute_error_runs():
    X = np.array(
        [
            [0.0],
            [1.0],
            [2.0],
            [3.0],
        ]
    )
    y = np.array([0.0, 1.0, 2.0, 3.0])

    model = DecisionTreeRegressor(criterion="absolute_error")
    model.fit(X, y)

    assert model.predict(X).shape == y.shape


def test_decision_tree_regressor_feature_importances():
    X = np.array(
        [
            [0.0, 10.0],
            [1.0, 10.0],
            [2.0, 10.0],
            [3.0, 10.0],
        ]
    )
    y = np.array([1.0, 1.0, 5.0, 5.0])

    model = DecisionTreeRegressor(max_depth=1)
    model.fit(X, y)

    assert model.feature_importances_.shape == (2,)
    assert model.feature_importances_[0] > model.feature_importances_[1]
    assert np.sum(model.feature_importances_) == pytest.approx(1.0)


def test_decision_tree_regressor_predict_before_fit_raises_error():
    model = DecisionTreeRegressor()

    with pytest.raises(ValueError):
        model.predict(np.array([[1.0]]))


def test_decision_tree_regressor_rejects_invalid_criterion():
    with pytest.raises(ValueError):
        DecisionTreeRegressor(criterion="bad")


def test_decision_tree_rejects_shape_mismatch():
    X = np.array([[0.0], [1.0]])
    y = np.array([0.0, 1.0, 2.0])

    model = DecisionTreeRegressor()

    with pytest.raises(ValueError):
        model.fit(X, y)


def test_decision_tree_max_features_runs():
    X = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [2.0, 1.0, 0.0],
            [3.0, 1.0, 0.0],
        ]
    )
    y = np.array([0, 0, 1, 1])

    model = DecisionTreeClassifier(max_features="sqrt", random_state=438)
    model.fit(X, y)

    assert model.predict(X).shape == y.shape