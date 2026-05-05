import numpy as np
import pytest

from jiayi_ml.preprocessing import MinMaxScaler, StandardScaler


def test_standard_scaler_transform_before_fit_raises_error():
    scaler = StandardScaler()

    with pytest.raises(ValueError):
        scaler.transform(np.array([[1.0, 2.0]]))


def test_standard_scaler_inverse_transform_before_fit_raises_error():
    scaler = StandardScaler()

    with pytest.raises(ValueError):
        scaler.inverse_transform(np.array([[1.0, 2.0]]))


def test_minmax_scaler_transform_before_fit_raises_error():
    scaler = MinMaxScaler()

    with pytest.raises(ValueError):
        scaler.transform(np.array([[1.0, 2.0]]))


def test_minmax_scaler_inverse_transform_before_fit_raises_error():
    scaler = MinMaxScaler()

    with pytest.raises(ValueError):
        scaler.inverse_transform(np.array([[1.0, 2.0]]))


def test_standard_scaler_rejects_empty_input_on_fit():
    scaler = StandardScaler()

    with pytest.raises(ValueError):
        scaler.fit(np.empty((0, 2)))


def test_minmax_scaler_rejects_empty_input_on_fit():
    scaler = MinMaxScaler()

    with pytest.raises(ValueError):
        scaler.fit(np.empty((0, 2)))


def test_standard_scaler_rejects_wrong_number_of_features_on_transform():
    scaler = StandardScaler()
    scaler.fit(np.array([[1.0, 2.0], [3.0, 4.0]]))

    with pytest.raises(ValueError):
        scaler.transform(np.array([[1.0, 2.0, 3.0]]))


def test_minmax_scaler_rejects_wrong_number_of_features_on_transform():
    scaler = MinMaxScaler()
    scaler.fit(np.array([[1.0, 2.0], [3.0, 4.0]]))

    with pytest.raises(ValueError):
        scaler.transform(np.array([[1.0, 2.0, 3.0]]))