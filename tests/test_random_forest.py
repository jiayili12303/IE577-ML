import numpy as np
import pytest

from jiayi_ml.supervised import RandomForestClassifier, RandomForestRegressor


def test_random_forest_classifier_fits_simple_binary_data():
    X = np.array(
        [
            [0.0],
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
        ]
    )
    y = np.array([0, 0, 0, 1, 1, 1])

    model = RandomForestClassifier(
        n_estimators=10,
        max_depth=2,
        bootstrap=False,
        random_state=438,
    )
    model.fit(X, y)

    predictions = model.predict(X)

    np.testing.assert_array_equal(predictions, y)
    assert model.score(X, y) == pytest.approx(1.0)


def test_random_forest_classifier_prediction_shape():
    X = np.array(
        [
            [0.0],
            [1.0],
            [2.0],
            [3.0],
        ]
    )
    y = np.array(["A", "A", "B", "B"])

    model = RandomForestClassifier(
        n_estimators=5,
        max_depth=2,
        bootstrap=False,
        random_state=438,
    )
    model.fit(X, y)

    predictions = model.predict(np.array([[0.5], [2.5]]))

    assert predictions.shape == (2,)


def test_random_forest_classifier_feature_importances():
    X = np.array(
        [
            [0.0, 10.0],
            [1.0, 10.0],
            [2.0, 10.0],
            [3.0, 10.0],
            [4.0, 10.0],
            [5.0, 10.0],
        ]
    )
    y = np.array([0, 0, 0, 1, 1, 1])

    model = RandomForestClassifier(
        n_estimators=5,
        max_depth=1,
        bootstrap=False,
        random_state=438,
    )
    model.fit(X, y)

    assert model.feature_importances_.shape == (2,)
    assert model.feature_importances_[0] > model.feature_importances_[1]
    assert np.sum(model.feature_importances_) == pytest.approx(1.0)


def test_random_forest_classifier_predict_before_fit_raises_error():
    model = RandomForestClassifier(n_estimators=3)

    with pytest.raises(ValueError):
        model.predict(np.array([[1.0]]))


def test_random_forest_classifier_rejects_invalid_n_estimators():
    with pytest.raises(ValueError):
        RandomForestClassifier(n_estimators=0)


def test_random_forest_classifier_rejects_wrong_number_of_features():
    X = np.array(
        [
            [0.0, 1.0],
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
        ]
    )
    y = np.array([0, 0, 1, 1])

    model = RandomForestClassifier(
        n_estimators=3,
        bootstrap=False,
        random_state=438,
    )
    model.fit(X, y)

    with pytest.raises(ValueError):
        model.predict(np.array([[1.0, 2.0, 3.0]]))


def test_random_forest_regressor_fits_simple_step_function():
    X = np.array(
        [
            [0.0],
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
        ]
    )
    y = np.array([1.0, 1.0, 1.0, 5.0, 5.0, 5.0])

    model = RandomForestRegressor(
        n_estimators=10,
        max_depth=2,
        bootstrap=False,
        random_state=438,
    )
    model.fit(X, y)

    predictions = model.predict(X)

    np.testing.assert_allclose(predictions, y)
    assert model.score(X, y) == pytest.approx(1.0)


def test_random_forest_regressor_prediction_shape():
    X = np.array(
        [
            [0.0],
            [1.0],
            [2.0],
            [3.0],
        ]
    )
    y = np.array([0.0, 1.0, 2.0, 3.0])

    model = RandomForestRegressor(
        n_estimators=5,
        max_depth=2,
        bootstrap=False,
        random_state=438,
    )
    model.fit(X, y)

    predictions = model.predict(np.array([[0.5], [2.5]]))

    assert predictions.shape == (2,)


def test_random_forest_regressor_feature_importances():
    X = np.array(
        [
            [0.0, 10.0],
            [1.0, 10.0],
            [2.0, 10.0],
            [3.0, 10.0],
            [4.0, 10.0],
            [5.0, 10.0],
        ]
    )
    y = np.array([1.0, 1.0, 1.0, 5.0, 5.0, 5.0])

    model = RandomForestRegressor(
        n_estimators=5,
        max_depth=1,
        bootstrap=False,
        random_state=438,
    )
    model.fit(X, y)

    assert model.feature_importances_.shape == (2,)
    assert model.feature_importances_[0] > model.feature_importances_[1]
    assert np.sum(model.feature_importances_) == pytest.approx(1.0)


def test_random_forest_regressor_predict_before_fit_raises_error():
    model = RandomForestRegressor(n_estimators=3)

    with pytest.raises(ValueError):
        model.predict(np.array([[1.0]]))


def test_random_forest_regressor_rejects_invalid_n_estimators():
    with pytest.raises(ValueError):
        RandomForestRegressor(n_estimators=0)


def test_random_forest_regressor_rejects_shape_mismatch():
    X = np.array([[0.0], [1.0]])
    y = np.array([0.0, 1.0, 2.0])

    model = RandomForestRegressor(n_estimators=3)

    with pytest.raises(ValueError):
        model.fit(X, y)


def test_random_forest_bootstrap_indices_shape():
    X = np.array(
        [
            [0.0],
            [1.0],
            [2.0],
            [3.0],
        ]
    )
    y = np.array([0, 0, 1, 1])

    model = RandomForestClassifier(
        n_estimators=4,
        bootstrap=True,
        random_state=438,
    )
    model.fit(X, y)

    assert len(model.bootstrap_indices_) == 4
    assert all(indices.shape == (4,) for indices in model.bootstrap_indices_)