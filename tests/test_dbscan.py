import numpy as np
import pytest

from jiayi_ml.unsupervised import DBSCAN


def test_dbscan_finds_two_simple_clusters():
    X = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [10.0, 10.0],
            [10.1, 10.0],
            [10.0, 10.1],
        ]
    )

    model = DBSCAN(eps=0.3, min_samples=2)
    labels = model.fit_predict(X)

    assert labels.shape == (6,)
    assert len(set(labels)) == 2
    assert -1 not in labels

    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4] == labels[5]
    assert labels[0] != labels[3]


def test_dbscan_identifies_noise_points():
    X = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [10.0, 10.0],
        ]
    )

    model = DBSCAN(eps=0.3, min_samples=2)
    labels = model.fit_predict(X)

    assert labels[-1] == -1


def test_dbscan_core_sample_indices_and_components():
    X = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [5.0, 5.0],
        ]
    )

    model = DBSCAN(eps=0.3, min_samples=2)
    model.fit(X)

    assert model.core_sample_indices_.ndim == 1
    assert model.components_.shape[1] == 2
    assert len(model.core_sample_indices_) >= 3


def test_dbscan_supports_manhattan_metric():
    X = np.array(
        [
            [0.0, 0.0],
            [0.2, 0.0],
            [5.0, 5.0],
            [5.2, 5.0],
        ]
    )

    model = DBSCAN(eps=0.3, min_samples=2, metric="manhattan")
    labels = model.fit_predict(X)

    assert len(set(labels)) == 2
    assert -1 not in labels


def test_dbscan_all_noise_when_eps_too_small():
    X = np.array(
        [
            [0.0],
            [1.0],
            [2.0],
        ]
    )

    model = DBSCAN(eps=0.1, min_samples=2)
    labels = model.fit_predict(X)

    np.testing.assert_array_equal(labels, np.array([-1, -1, -1]))


def test_dbscan_rejects_invalid_eps():
    with pytest.raises(ValueError):
        DBSCAN(eps=0.0)


def test_dbscan_rejects_invalid_min_samples():
    with pytest.raises(ValueError):
        DBSCAN(min_samples=0)


def test_dbscan_rejects_invalid_metric():
    with pytest.raises(ValueError):
        DBSCAN(metric="bad_metric")