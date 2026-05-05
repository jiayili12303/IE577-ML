import numpy as np
import pytest

from jiayi_ml.preprocessing import MinMaxScaler, StandardScaler


def test_standard_scaler_fit_transform_mean_and_std():
    X = np.array(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
        ]
    )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    np.testing.assert_allclose(np.mean(X_scaled, axis=0), np.array([0.0, 0.0]))
    np.testing.assert_allclose(np.std(X_scaled, axis=0), np.array([1.0, 1.0]))


def test_standard_scaler_inverse_transform():
    X = np.array(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
        ]
    )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_recovered = scaler.inverse_transform(X_scaled)

    np.testing.assert_allclose(X_recovered, X)


def test_standard_scaler_constant_feature():
    X = np.array(
        [
            [1.0, 5.0],
            [2.0, 5.0],
            [3.0, 5.0],
        ]
    )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    assert not np.isnan(X_scaled).any()
    np.testing.assert_allclose(X_scaled[:, 1], np.array([0.0, 0.0, 0.0]))


def test_standard_scaler_transform_before_fit_raises_error():
    scaler = StandardScaler()

    with pytest.raises(ValueError):
        scaler.transform(np.array([[1.0, 2.0]]))


def test_minmax_scaler_fit_transform_default_range():
    X = np.array(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
        ]
    )

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    np.testing.assert_allclose(np.min(X_scaled, axis=0), np.array([0.0, 0.0]))
    np.testing.assert_allclose(np.max(X_scaled, axis=0), np.array([1.0, 1.0]))


def test_minmax_scaler_custom_range():
    X = np.array(
        [
            [1.0],
            [2.0],
            [3.0],
        ]
    )

    scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
    X_scaled = scaler.fit_transform(X)

    np.testing.assert_allclose(np.min(X_scaled), -1.0)
    np.testing.assert_allclose(np.max(X_scaled), 1.0)


def test_minmax_scaler_inverse_transform():
    X = np.array(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
        ]
    )

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_recovered = scaler.inverse_transform(X_scaled)

    np.testing.assert_allclose(X_recovered, X)


def test_minmax_scaler_invalid_feature_range_raises_error():
    with pytest.raises(ValueError):
        MinMaxScaler(feature_range=(1.0, 0.0))


def test_scaler_rejects_wrong_number_of_features():
    X_train = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    X_test = np.array(
        [
            [1.0, 2.0, 3.0],
        ]
    )

    scaler = StandardScaler()
    scaler.fit(X_train)

    with pytest.raises(ValueError):
        scaler.transform(X_test)