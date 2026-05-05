import numpy as np
import pytest

from jiayi_ml.unsupervised import KMeans


def test_kmeans_finds_simple_clusters():
    X = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [10.0, 10.0],
            [10.1, 10.0],
        ]
    )

    model = KMeans(n_clusters=2, random_state=438, n_init=5)
    model.fit(X)

    assert model.cluster_centers_.shape == (2, 2)
    assert model.labels_.shape == (4,)
    assert model.inertia_ < 0.05

    sorted_centers = model.cluster_centers_[np.argsort(model.cluster_centers_[:, 0])]
    np.testing.assert_allclose(
        sorted_centers,
        np.array(
            [
                [0.05, 0.0],
                [10.05, 10.0],
            ]
        ),
        atol=1e-2,
    )


def test_kmeans_fit_predict_returns_labels():
    X = np.array(
        [
            [0.0],
            [0.2],
            [5.0],
            [5.2],
        ]
    )

    model = KMeans(n_clusters=2, random_state=438, n_init=5)
    labels = model.fit_predict(X)

    assert labels.shape == (4,)
    assert len(np.unique(labels)) == 2


def test_kmeans_predict_assigns_nearest_cluster():
    X_train = np.array(
        [
            [0.0],
            [0.2],
            [10.0],
            [10.2],
        ]
    )

    model = KMeans(n_clusters=2, random_state=438, n_init=5)
    model.fit(X_train)

    predictions = model.predict(np.array([[0.1], [10.1]]))

    assert predictions.shape == (2,)
    assert predictions[0] != predictions[1]


def test_kmeans_transform_shape_and_nonnegative_distances():
    X = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [10.0, 10.0],
        ]
    )

    model = KMeans(n_clusters=2, random_state=438, n_init=5)
    model.fit(X)

    distances = model.transform(X)

    assert distances.shape == (3, 2)
    assert np.all(distances >= 0.0)


def test_kmeans_predict_before_fit_raises_error():
    model = KMeans(n_clusters=2)

    with pytest.raises(ValueError):
        model.predict(np.array([[1.0, 2.0]]))


def test_kmeans_rejects_invalid_number_of_clusters():
    with pytest.raises(ValueError):
        KMeans(n_clusters=0)


def test_kmeans_rejects_too_many_clusters():
    X = np.array(
        [
            [0.0],
            [1.0],
        ]
    )

    model = KMeans(n_clusters=3)

    with pytest.raises(ValueError):
        model.fit(X)


def test_kmeans_rejects_invalid_init():
    with pytest.raises(ValueError):
        KMeans(init="bad_init")


def test_kmeans_rejects_wrong_number_of_features():
    X_train = np.array(
        [
            [0.0, 1.0],
            [1.0, 2.0],
            [10.0, 11.0],
        ]
    )
    X_test = np.array([[1.0, 2.0, 3.0]])

    model = KMeans(n_clusters=2, random_state=438)
    model.fit(X_train)

    with pytest.raises(ValueError):
        model.predict(X_test)