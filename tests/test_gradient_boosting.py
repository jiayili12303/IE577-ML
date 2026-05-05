import numpy as np
import pytest

from jiayi_ml.supervised import GradientBoostingRegressor


def test_gradient_boosting_regressor_fits_simple_signal():
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])

    model = GradientBoostingRegressor(
        n_estimators=20,
        learning_rate=0.2,
        max_depth=2,
    )
    model.fit(X, y)

    predictions = model.predict(X)

    assert predictions.shape == y.shape
    assert np.mean((predictions - y) ** 2) < 0.5


def test_gradient_boosting_train_loss_decreases():
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])

    model = GradientBoostingRegressor(
        n_estimators=10,
        learning_rate=0.2,
        max_depth=1,
    )
    model.fit(X, y)

    assert len(model.train_loss_) == 10
    assert model.train_loss_[-1] < model.train_loss_[0]


def test_gradient_boosting_staged_predict_outputs_all_stages():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 1.0, 2.0, 3.0])

    model = GradientBoostingRegressor(n_estimators=5, learning_rate=0.1)
    model.fit(X, y)

    staged_predictions = list(model.staged_predict(X))

    assert len(staged_predictions) == 5
    for prediction in staged_predictions:
        assert prediction.shape == y.shape


def test_gradient_boosting_predict_before_fit_raises_error():
    model = GradientBoostingRegressor()

    with pytest.raises(ValueError):
        model.predict(np.array([[1.0]]))


def test_gradient_boosting_rejects_invalid_n_estimators():
    with pytest.raises(ValueError):
        GradientBoostingRegressor(n_estimators=0)


def test_gradient_boosting_rejects_invalid_learning_rate():
    with pytest.raises(ValueError):
        GradientBoostingRegressor(learning_rate=0.0)


def test_gradient_boosting_rejects_invalid_max_depth():
    with pytest.raises(ValueError):
        GradientBoostingRegressor(max_depth=0)


def test_gradient_boosting_rejects_invalid_min_samples_split():
    with pytest.raises(ValueError):
        GradientBoostingRegressor(min_samples_split=1)


def test_gradient_boosting_rejects_invalid_min_samples_leaf():
    with pytest.raises(ValueError):
        GradientBoostingRegressor(min_samples_leaf=0)


def test_gradient_boosting_rejects_shape_mismatch():
    X = np.array([[0.0], [1.0]])
    y = np.array([0.0, 1.0, 2.0])

    model = GradientBoostingRegressor()

    with pytest.raises(ValueError):
        model.fit(X, y)


def test_gradient_boosting_rejects_empty_input():
    model = GradientBoostingRegressor()

    with pytest.raises(ValueError):
        model.fit(np.empty((0, 2)), np.array([]))


def test_gradient_boosting_rejects_wrong_number_of_features():
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([0.0, 1.0, 2.0])

    model = GradientBoostingRegressor()
    model.fit(X, y)

    with pytest.raises(ValueError):
        model.predict(np.array([[1.0, 2.0]]))


def test_gradient_boosting_score_returns_float():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 1.0, 2.0, 3.0])

    model = GradientBoostingRegressor(n_estimators=10, learning_rate=0.2)
    model.fit(X, y)

    score = model.score(X, y)

    assert isinstance(score, float)