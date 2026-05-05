import numpy as np
import pytest

from jiayi_ml.supervised import LassoRegression, LinearRegression, RidgeRegression


def test_linear_regression_fits_simple_line():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([1.0, 3.0, 5.0, 7.0])

    model = LinearRegression()
    model.fit(X, y)

    assert model.intercept_ == pytest.approx(1.0)
    np.testing.assert_allclose(model.coef_, np.array([2.0]))

    predictions = model.predict(X)
    np.testing.assert_allclose(predictions, y)


def test_linear_regression_without_intercept():
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([2.0, 4.0, 6.0])

    model = LinearRegression(fit_intercept=False)
    model.fit(X, y)

    assert model.intercept_ == pytest.approx(0.0)
    np.testing.assert_allclose(model.coef_, np.array([2.0]))


def test_linear_regression_score_perfect_fit():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([1.0, 3.0, 5.0, 7.0])

    model = LinearRegression()
    model.fit(X, y)

    assert model.score(X, y) == pytest.approx(1.0)


def test_linear_regression_predict_before_fit_raises_error():
    model = LinearRegression()

    with pytest.raises(ValueError):
        model.predict(np.array([[1.0]]))


def test_linear_regression_rejects_shape_mismatch():
    X = np.array([[1.0], [2.0]])
    y = np.array([1.0, 2.0, 3.0])

    model = LinearRegression()

    with pytest.raises(ValueError):
        model.fit(X, y)


def test_linear_regression_rejects_wrong_number_of_features():
    X_train = np.array([[1.0, 2.0], [3.0, 4.0]])
    y_train = np.array([1.0, 2.0])
    X_test = np.array([[1.0, 2.0, 3.0]])

    model = LinearRegression()
    model.fit(X_train, y_train)

    with pytest.raises(ValueError):
        model.predict(X_test)


def test_ridge_regression_alpha_zero_matches_linear_regression():
    X = np.array(
        [
            [1.0, 2.0],
            [2.0, 1.0],
            [3.0, 4.0],
            [4.0, 3.0],
        ]
    )
    y = np.array([3.0, 3.0, 7.0, 7.0])

    linear = LinearRegression()
    ridge = RidgeRegression(alpha=0.0)

    linear.fit(X, y)
    ridge.fit(X, y)

    np.testing.assert_allclose(ridge.predict(X), linear.predict(X), atol=1e-8)


def test_ridge_regression_shrinks_coefficients():
    rng = np.random.default_rng(438)
    X = rng.normal(size=(100, 3))
    y = 5.0 * X[:, 0] + 0.1 * rng.normal(size=100)

    linear = LinearRegression()
    ridge = RidgeRegression(alpha=50.0)

    linear.fit(X, y)
    ridge.fit(X, y)

    assert np.linalg.norm(ridge.coef_) < np.linalg.norm(linear.coef_)


def test_ridge_regression_rejects_negative_alpha():
    with pytest.raises(ValueError):
        RidgeRegression(alpha=-1.0)


def test_lasso_regression_prediction_shape():
    rng = np.random.default_rng(438)
    X = rng.normal(size=(50, 4))
    y = 2.0 * X[:, 0] - 1.0 * X[:, 1] + rng.normal(scale=0.1, size=50)

    model = LassoRegression(alpha=0.01, max_iter=5000, tol=1e-8)
    model.fit(X, y)

    predictions = model.predict(X)

    assert predictions.shape == y.shape


def test_lasso_regression_identifies_relevant_feature():
    rng = np.random.default_rng(438)
    X = rng.normal(size=(120, 3))
    y = 3.0 * X[:, 0] + rng.normal(scale=0.05, size=120)

    model = LassoRegression(alpha=0.05, max_iter=10000, tol=1e-8)
    model.fit(X, y)

    assert abs(model.coef_[0]) > 2.5
    assert abs(model.coef_[1]) < 0.2
    assert abs(model.coef_[2]) < 0.2


def test_lasso_regression_rejects_negative_alpha():
    with pytest.raises(ValueError):
        LassoRegression(alpha=-0.1)


def test_lasso_regression_predict_before_fit_raises_error():
    model = LassoRegression()

    with pytest.raises(ValueError):
        model.predict(np.array([[1.0, 2.0]]))