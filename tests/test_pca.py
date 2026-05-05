import numpy as np
import pytest

from jiayi_ml.unsupervised import PCA


def test_pca_fit_transform_shape():
    X = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0],
        ]
    )

    model = PCA(n_components=2)
    X_transformed = model.fit_transform(X)

    assert X_transformed.shape == (4, 2)
    assert model.components_.shape == (2, 3)
    assert model.explained_variance_.shape == (2,)
    assert model.explained_variance_ratio_.shape == (2,)


def test_pca_explained_variance_ratio_sums_to_one_when_all_components_kept():
    X = np.array(
        [
            [1.0, 2.0],
            [2.0, 1.0],
            [3.0, 4.0],
            [4.0, 3.0],
        ]
    )

    model = PCA()
    model.fit(X)

    assert np.sum(model.explained_variance_ratio_) == pytest.approx(1.0)


def test_pca_first_component_captures_main_variation():
    X = np.array(
        [
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
        ]
    )

    model = PCA(n_components=1)
    model.fit(X)

    assert model.explained_variance_ratio_[0] == pytest.approx(1.0)


def test_pca_inverse_transform_full_components_recovers_data():
    X = np.array(
        [
            [1.0, 2.0],
            [2.0, 1.0],
            [3.0, 4.0],
            [4.0, 3.0],
        ]
    )

    model = PCA(n_components=2)
    X_transformed = model.fit_transform(X)
    X_recovered = model.inverse_transform(X_transformed)

    np.testing.assert_allclose(X_recovered, X)


def test_pca_transform_before_fit_raises_error():
    model = PCA(n_components=2)

    with pytest.raises(ValueError):
        model.transform(np.array([[1.0, 2.0]]))


def test_pca_rejects_invalid_n_components_type():
    with pytest.raises(ValueError):
        PCA(n_components=1.5)


def test_pca_rejects_negative_n_components():
    with pytest.raises(ValueError):
        PCA(n_components=-1)


def test_pca_rejects_too_many_components():
    X = np.array(
        [
            [1.0, 2.0],
            [2.0, 3.0],
        ]
    )

    model = PCA(n_components=3)

    with pytest.raises(ValueError):
        model.fit(X)


def test_pca_rejects_wrong_number_of_features():
    X_train = np.array(
        [
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
        ]
    )

    X_test = np.array([[1.0, 2.0, 3.0]])

    model = PCA(n_components=1)
    model.fit(X_train)

    with pytest.raises(ValueError):
        model.transform(X_test)


def test_pca_inverse_transform_rejects_wrong_number_of_components():
    X = np.array(
        [
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
        ]
    )

    model = PCA(n_components=1)
    model.fit(X)

    with pytest.raises(ValueError):
        model.inverse_transform(np.array([[1.0, 2.0]]))