"""
Feature scaling utilities implemented from scratch using NumPy.

This module provides two common preprocessing classes:

- StandardScaler: standardizes features by removing the mean and scaling to unit variance.
- MinMaxScaler: rescales features to a fixed range, usually [0, 1].

These classes follow a scikit-learn-like API with fit, transform, fit_transform,
and inverse_transform methods.
"""

from __future__ import annotations

import numpy as np


def _validate_2d_array(X: np.ndarray) -> np.ndarray:
    """
    Convert input data to a 2D NumPy array of type float.

    Parameters
    ----------
    X : array-like
        Input feature matrix.

    Returns
    -------
    np.ndarray
        A 2D NumPy array with dtype float.

    Raises
    ------
    ValueError
        If the input array is empty or has more than two dimensions.
    """
    X = np.asarray(X, dtype=float)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    if X.ndim != 2:
        raise ValueError("Input data must be a 1D or 2D array.")

    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("Input data must not be empty.")

    return X


class StandardScaler:
    """
    Standardize features by removing the mean and scaling to unit variance.

    For each feature, the transformed value is:

        z = (x - mean) / standard_deviation

    If a feature has zero variance, its scale is set to 1 to avoid division by zero.

    Attributes
    ----------
    mean_ : np.ndarray
        Mean of each feature in the training data.
    var_ : np.ndarray
        Variance of each feature in the training data.
    scale_ : np.ndarray
        Standard deviation of each feature, with zero values replaced by 1.
    n_features_in_ : int
        Number of features seen during fitting.
    """

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.var_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.n_features_in_: int | None = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        """
        Compute the mean and standard deviation for each feature.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        StandardScaler
            Fitted scaler.
        """
        X = _validate_2d_array(X)

        self.mean_ = np.mean(X, axis=0)
        self.var_ = np.var(X, axis=0)
        self.scale_ = np.sqrt(self.var_)

        self.scale_ = np.where(self.scale_ == 0, 1.0, self.scale_)
        self.n_features_in_ = X.shape[1]

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Standardize data using the fitted mean and standard deviation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        np.ndarray
            Standardized data.

        Raises
        ------
        ValueError
            If the scaler has not been fitted or feature dimensions do not match.
        """
        self._check_is_fitted()
        X = _validate_2d_array(X)
        self._check_n_features(X)

        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit the scaler and transform the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        np.ndarray
            Standardized data.
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Convert standardized data back to the original scale.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Standardized data.

        Returns
        -------
        np.ndarray
            Data in the original scale.
        """
        self._check_is_fitted()
        X = _validate_2d_array(X)
        self._check_n_features(X)

        return X * self.scale_ + self.mean_

    def _check_is_fitted(self) -> None:
        """Raise an error if the scaler has not been fitted."""
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("This StandardScaler instance is not fitted yet.")

    def _check_n_features(self, X: np.ndarray) -> None:
        """Raise an error if the number of features does not match training data."""
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, but got {X.shape[1]}."
            )


class MinMaxScaler:
    """
    Rescale features to a fixed range.

    For each feature, the transformed value is:

        x_scaled = (x - min) / (max - min) * (range_max - range_min) + range_min

    If a feature has zero range, its data range is set to 1 to avoid division by zero.

    Parameters
    ----------
    feature_range : tuple of float, default=(0.0, 1.0)
        Desired range of transformed data.

    Attributes
    ----------
    data_min_ : np.ndarray
        Minimum value of each feature in the training data.
    data_max_ : np.ndarray
        Maximum value of each feature in the training data.
    data_range_ : np.ndarray
        Range of each feature, with zero ranges replaced by 1.
    n_features_in_ : int
        Number of features seen during fitting.
    """

    def __init__(self, feature_range: tuple[float, float] = (0.0, 1.0)) -> None:
        if len(feature_range) != 2:
            raise ValueError("feature_range must be a tuple of two values.")

        if feature_range[0] >= feature_range[1]:
            raise ValueError("feature_range minimum must be less than maximum.")

        self.feature_range = feature_range
        self.data_min_: np.ndarray | None = None
        self.data_max_: np.ndarray | None = None
        self.data_range_: np.ndarray | None = None
        self.n_features_in_: int | None = None

    def fit(self, X: np.ndarray) -> "MinMaxScaler":
        """
        Compute the minimum and maximum values for each feature.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        MinMaxScaler
            Fitted scaler.
        """
        X = _validate_2d_array(X)

        self.data_min_ = np.min(X, axis=0)
        self.data_max_ = np.max(X, axis=0)
        self.data_range_ = self.data_max_ - self.data_min_
        self.data_range_ = np.where(self.data_range_ == 0, 1.0, self.data_range_)
        self.n_features_in_ = X.shape[1]

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Rescale data using the fitted minimum and maximum values.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        np.ndarray
            Rescaled data.
        """
        self._check_is_fitted()
        X = _validate_2d_array(X)
        self._check_n_features(X)

        range_min, range_max = self.feature_range
        X_std = (X - self.data_min_) / self.data_range_

        return X_std * (range_max - range_min) + range_min

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit the scaler and transform the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        np.ndarray
            Rescaled data.
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Convert rescaled data back to the original scale.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Rescaled data.

        Returns
        -------
        np.ndarray
            Data in the original scale.
        """
        self._check_is_fitted()
        X = _validate_2d_array(X)
        self._check_n_features(X)

        range_min, range_max = self.feature_range
        X_std = (X - range_min) / (range_max - range_min)

        return X_std * self.data_range_ + self.data_min_

    def _check_is_fitted(self) -> None:
        """Raise an error if the scaler has not been fitted."""
        if self.data_min_ is None or self.data_max_ is None or self.data_range_ is None:
            raise ValueError("This MinMaxScaler instance is not fitted yet.")

    def _check_n_features(self, X: np.ndarray) -> None:
        """Raise an error if the number of features does not match training data."""
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, but got {X.shape[1]}."
            )