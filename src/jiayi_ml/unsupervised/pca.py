"""
Principal Component Analysis implemented from scratch using NumPy.

This module provides a PCA class with a scikit-learn-like API. PCA projects
high-dimensional data onto orthogonal directions that capture maximum variance.
"""

from __future__ import annotations

import numpy as np


def _validate_X(X):
    """
    Convert input features to a 2D NumPy array of type float.

    Parameters
    ----------
    X : array-like
        Feature matrix.

    Returns
    -------
    np.ndarray
        Validated feature matrix.
    """
    X = np.asarray(X, dtype=float)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    if X.ndim != 2:
        raise ValueError("X must be a 1D or 2D array.")

    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X must not be empty.")

    return X


class PCA:
    """
    Principal Component Analysis implemented from scratch.

    Parameters
    ----------
    n_components : int or None, default=None
        Number of principal components to keep. If None, all components are kept.

    Attributes
    ----------
    components_ : np.ndarray
        Principal axes in feature space, with shape (n_components, n_features).
    explained_variance_ : np.ndarray
        Variance explained by each selected principal component.
    explained_variance_ratio_ : np.ndarray
        Proportion of total variance explained by each selected component.
    singular_values_ : np.ndarray
        Singular values corresponding to selected components.
    mean_ : np.ndarray
        Per-feature mean estimated from the training data.
    n_components_ : int
        Actual number of components retained.
    n_features_in_ : int
        Number of features seen during fitting.
    """

    def __init__(self, n_components=None):
        if n_components is not None:
            if not isinstance(n_components, int):
                raise ValueError("n_components must be an integer or None.")
            if n_components <= 0:
                raise ValueError("n_components must be positive.")

        self.n_components = n_components

        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.singular_values_ = None
        self.mean_ = None
        self.n_components_ = None
        self.n_features_in_ = None

    def fit(self, X):
        """
        Fit PCA on the input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        PCA
            Fitted PCA model.
        """
        X = _validate_X(X)
        n_samples, n_features = X.shape

        if self.n_components is None:
            n_components = min(n_samples, n_features)
        else:
            n_components = self.n_components

        if n_components > min(n_samples, n_features):
            raise ValueError(
                "n_components cannot be larger than min(n_samples, n_features)."
            )

        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        _, singular_values, Vt = np.linalg.svd(X_centered, full_matrices=False)

        explained_variance = (singular_values ** 2) / (n_samples - 1)

        total_variance = np.sum(explained_variance)
        if total_variance == 0:
            explained_variance_ratio = np.zeros_like(explained_variance)
        else:
            explained_variance_ratio = explained_variance / total_variance

        self.components_ = Vt[:n_components]
        self.explained_variance_ = explained_variance[:n_components]
        self.explained_variance_ratio_ = explained_variance_ratio[:n_components]
        self.singular_values_ = singular_values[:n_components]
        self.n_components_ = n_components
        self.n_features_in_ = n_features

        return self

    def transform(self, X):
        """
        Project data onto the principal components.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        np.ndarray
            Transformed data of shape (n_samples, n_components).
        """
        self._check_is_fitted()
        X = _validate_X(X)
        self._check_n_features(X)

        X_centered = X - self.mean_
        return X_centered @ self.components_.T

    def fit_transform(self, X):
        """
        Fit PCA and project data onto the principal components.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        np.ndarray
            Transformed data.
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X_transformed):
        """
        Transform projected data back to the original feature space.

        Parameters
        ----------
        X_transformed : array-like of shape (n_samples, n_components)
            Data in principal component space.

        Returns
        -------
        np.ndarray
            Reconstructed data in original feature space.
        """
        self._check_is_fitted()
        X_transformed = np.asarray(X_transformed, dtype=float)

        if X_transformed.ndim == 1:
            X_transformed = X_transformed.reshape(1, -1)

        if X_transformed.ndim != 2:
            raise ValueError("X_transformed must be a 1D or 2D array.")

        if X_transformed.shape[1] != self.n_components_:
            raise ValueError(
                f"Expected {self.n_components_} components, "
                f"but got {X_transformed.shape[1]}."
            )

        return X_transformed @ self.components_ + self.mean_

    def _check_is_fitted(self):
        """Raise an error if PCA has not been fitted."""
        if self.components_ is None or self.mean_ is None:
            raise ValueError("This PCA instance is not fitted yet.")

    def _check_n_features(self, X):
        """Raise an error if the number of features does not match training data."""
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, but got {X.shape[1]}."
            )