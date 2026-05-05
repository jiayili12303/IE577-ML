import numpy as np
import pytest

from jiayi_ml.unsupervised import AgglomerativeClustering


def test_agglomerative_finds_two_simple_clusters():
    X = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [10.0, 10.0],
            [10.1, 10.0],
        ]
    )

    model = AgglomerativeClustering(n_clusters=2, linkage="average")
    labels = model.fit_predict(X)

    assert labels.shape == (4,)
    assert len(np.unique(labels)) == 2

    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]


def test_agglomerative_children_and_distances_shapes():
    X = np.array(
        [
            [0.0],
            [0.1],
            [5.0],
            [5.1],
        ]
    )

    model = AgglomerativeClustering(n_clusters=2)
    model.fit(X)

    assert model.children_.shape == (2, 2)
    assert model.distances_.shape == (2,)


def test_agglomerative_single_linkage_runs():
    X = np.array(
        [
            [0.0],
            [0.2],
            [5.0],
            [5.2],
        ]
    )

    model = AgglomerativeClustering(n_clusters=2, linkage="single")
    labels = model.fit_predict(X)

    assert len(np.unique(labels)) == 2


def test_agglomerative_complete_linkage_runs():
    X = np.array(
        [
            [0.0],
            [0.2],
            [5.0],
            [5.2],
        ]
    )

    model = AgglomerativeClustering(n_clusters=2, linkage="complete")
    labels = model.fit_predict(X)

    assert len(np.unique(labels)) == 2


def test_agglomerative_supports_manhattan_metric():
    X = np.array(
        [
            [0.0, 0.0],
            [0.2, 0.0],
            [5.0, 5.0],
            [5.2, 5.0],
        ]
    )

    model = AgglomerativeClustering(
        n_clusters=2,
        linkage="average",
        metric="manhattan",
    )
    labels = model.fit_predict(X)

    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]


def test_agglomerative_one_cluster():
    X = np.array(
        [
            [0.0],
            [1.0],
            [2.0],
        ]
    )

    model = AgglomerativeClustering(n_clusters=1)
    labels = model.fit_predict(X)

    np.testing.assert_array_equal(labels, np.array([0, 0, 0]))
    assert model.children_.shape == (2, 2)


def test_agglomerative_rejects_invalid_n_clusters():
    with pytest.raises(ValueError):
        AgglomerativeClustering(n_clusters=0)


def test_agglomerative_rejects_too_many_clusters():
    X = np.array(
        [
            [0.0],
            [1.0],
        ]
    )

    model = AgglomerativeClustering(n_clusters=3)

    with pytest.raises(ValueError):
        model.fit(X)


def test_agglomerative_rejects_invalid_linkage():
    with pytest.raises(ValueError):
        AgglomerativeClustering(linkage="ward")


def test_agglomerative_rejects_invalid_metric():
    with pytest.raises(ValueError):
        AgglomerativeClustering(metric="cosine")