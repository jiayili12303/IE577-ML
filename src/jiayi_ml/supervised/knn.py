"""
K-nearest neighbors models implemented from scratch using NumPy.

This module includes:

- KNNClassifier: k-nearest neighbors for classification.
- KNNRegressor: k-nearest neighbors for regression.

Both classes use a lazy-learning approach: training stores the data, and
prediction is made by finding the nearest training samples.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from jiayi_ml.metrics import accuracy_score, r2_score


def _validate_X(X: np.ndarray) -> np.ndarray:
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


def _validate_X_y(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Validate feature matrix X and target vector y.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    y : array-like of shape (n_samples,)
        Target values or class labels.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Validated X and y.
    """
    X = _validate_X(X)
    y = np.asarray(y).ravel()

    if y.size == 0:
        raise ValueError("y must not be empty.")

    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must contain the same number of samples.")

    return X, y


def _compute_distances(
    X_train: np.ndarray,
    x: np.ndarray,
    metric: str,
) -> np.ndarray:
    """
    Compute distances between all training samples and one query sample.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix.
    x : np.ndarray
        Query sample.
    metric : {"euclidean", "manhattan"}
        Distance metric.

    Returns
    -------
    np.ndarray
        Distance from each training sample to the query sample.
    """
    if metric == "euclidean":
        return np.sqrt(np.sum((X_train - x) ** 2, axis=1))

    if metric == "manhattan":
        return np.sum(np.abs(X_train - x), axis=1)

    raise ValueError("metric must be either 'euclidean' or 'manhattan'.")


class KNNClassifier:
    """
    K-nearest neighbors classifier implemented from scratch.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors to use for prediction.
    metric : {"euclidean", "manhattan"}, default="euclidean"
        Distance metric used to identify nearest neighbors.
    weights : {"uniform", "distance"}, default="uniform"
        Weighting strategy. If "uniform", each neighbor votes equally. If
        "distance", closer neighbors receive more weight.

    Attributes
    ----------
    X_train_ : np.ndarray
        Stored training features.
    y_train_ : np.ndarray
        Stored training labels.
    classes_ : np.ndarray
        Unique class labels.
    n_features_in_ : int
        Number of features seen during fitting.
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        metric: str = "euclidean",
        weights: str = "uniform",
    ) -> None:
        if n_neighbors <= 0:
            raise ValueError("n_neighbors must be positive.")

        if metric not in {"euclidean", "manhattan"}:
            raise ValueError("metric must be either 'euclidean' or 'manhattan'.")

        if weights not in {"uniform", "distance"}:
            raise ValueError("weights must be either 'uniform' or 'distance'.")

        self.n_neighbors = n_neighbors
        self.metric = metric
        self.weights = weights

        self.X_train_: np.ndarray | None = None
        self.y_train_: np.ndarray | None = None
        self.classes_: np.ndarray | None = None
        self.n_features_in_: int | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNClassifier":
        """
        Store the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Class labels.

        Returns
        -------
        KNNClassifier
            Fitted classifier.
        """
        X, y = _validate_X_y(X, y)

        if self.n_neighbors > X.shape[0]:
            raise ValueError("n_neighbors cannot be larger than number of samples.")

        self.X_train_ = X
        self.y_train_ = y
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for input samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features.

        Returns
        -------
        np.ndarray
            Predicted class labels.
        """
        self._check_is_fitted()
        X = _validate_X(X)
        self._check_n_features(X)

        return np.array([self._predict_one(x) for x in X])

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute classification accuracy on the given data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features.
        y : array-like of shape (n_samples,)
            True class labels.

        Returns
        -------
        float
            Accuracy score.
        """
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    def _predict_one(self, x: np.ndarray):
        """
        Predict the class label for one sample.

        Parameters
        ----------
        x : np.ndarray
            One input sample.

        Returns
        -------
        object
            Predicted class label.
        """
        distances = _compute_distances(self.X_train_, x, self.metric)
        neighbor_indices = np.argsort(distances)[: self.n_neighbors]
        neighbor_labels = self.y_train_[neighbor_indices]

        if self.weights == "uniform":
            counts = Counter(neighbor_labels)
            max_count = max(counts.values())
            tied_labels = [label for label, count in counts.items() if count == max_count]
            return sorted(tied_labels)[0]

        neighbor_distances = distances[neighbor_indices]
        label_weights = {label: 0.0 for label in self.classes_}

        for label, distance in zip(neighbor_labels, neighbor_distances):
            if distance == 0:
                return label
            label_weights[label] += 1.0 / distance

        max_weight = max(label_weights.values())
        tied_labels = [
            label for label, weight in label_weights.items() if weight == max_weight
        ]

        return sorted(tied_labels)[0]

    def _check_is_fitted(self) -> None:
        """Raise an error if the classifier has not been fitted."""
        if self.X_train_ is None or self.y_train_ is None:
            raise ValueError("This KNNClassifier instance is not fitted yet.")

    def _check_n_features(self, X: np.ndarray) -> None:
        """Raise an error if the number of features does not match training data."""
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, but got {X.shape[1]}."
            )


class KNNRegressor:
    """
    K-nearest neighbors regressor implemented from scratch.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors to use for prediction.
    metric : {"euclidean", "manhattan"}, default="euclidean"
        Distance metric used to identify nearest neighbors.
    weights : {"uniform", "distance"}, default="uniform"
        Weighting strategy. If "uniform", predictions are the average target
        value among neighbors. If "distance", closer neighbors receive more weight.

    Attributes
    ----------
    X_train_ : np.ndarray
        Stored training features.
    y_train_ : np.ndarray
        Stored training targets.
    n_features_in_ : int
        Number of features seen during fitting.
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        metric: str = "euclidean",
        weights: str = "uniform",
    ) -> None:
        if n_neighbors <= 0:
            raise ValueError("n_neighbors must be positive.")

        if metric not in {"euclidean", "manhattan"}:
            raise ValueError("metric must be either 'euclidean' or 'manhattan'.")

        if weights not in {"uniform", "distance"}:
            raise ValueError("weights must be either 'uniform' or 'distance'.")

        self.n_neighbors = n_neighbors
        self.metric = metric
        self.weights = weights

        self.X_train_: np.ndarray | None = None
        self.y_train_: np.ndarray | None = None
        self.n_features_in_: int | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNRegressor":
        """
        Store the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        KNNRegressor
            Fitted regressor.
        """
        X, y = _validate_X_y(X, y)
        y = y.astype(float)

        if self.n_neighbors > X.shape[0]:
            raise ValueError("n_neighbors cannot be larger than number of samples.")

        self.X_train_ = X
        self.y_train_ = y
        self.n_features_in_ = X.shape[1]

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values for input samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features.

        Returns
        -------
        np.ndarray
            Predicted target values.
        """
        self._check_is_fitted()
        X = _validate_X(X)
        self._check_n_features(X)

        return np.array([self._predict_one(x) for x in X], dtype=float)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the R-squared score on the given data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features.
        y : array-like of shape (n_samples,)
            True target values.

        Returns
        -------
        float
            R-squared score.
        """
        y_pred = self.predict(X)
        return r2_score(y, y_pred)

    def _predict_one(self, x: np.ndarray) -> float:
        """
        Predict the target value for one sample.

        Parameters
        ----------
        x : np.ndarray
            One input sample.

        Returns
        -------
        float
            Predicted target value.
        """
        distances = _compute_distances(self.X_train_, x, self.metric)
        neighbor_indices = np.argsort(distances)[: self.n_neighbors]
        neighbor_values = self.y_train_[neighbor_indices]

        if self.weights == "uniform":
            return float(np.mean(neighbor_values))

        neighbor_distances = distances[neighbor_indices]

        if np.any(neighbor_distances == 0):
            return float(np.mean(neighbor_values[neighbor_distances == 0]))

        inverse_distances = 1.0 / neighbor_distances
        weights = inverse_distances / np.sum(inverse_distances)

        return float(np.sum(weights * neighbor_values))

    def _check_is_fitted(self) -> None:
        """Raise an error if the regressor has not been fitted."""
        if self.X_train_ is None or self.y_train_ is None:
            raise ValueError("This KNNRegressor instance is not fitted yet.")

    def _check_n_features(self, X: np.ndarray) -> None:
        """Raise an error if the number of features does not match training data."""
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, but got {X.shape[1]}."
            )