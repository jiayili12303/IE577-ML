"""
Naive Bayes classifiers implemented from scratch using NumPy.

This module currently includes:

- GaussianNaiveBayes: Naive Bayes classifier for continuous features assuming
  a Gaussian class-conditional distribution.
"""

from __future__ import annotations

import numpy as np

from jiayi_ml.metrics import accuracy_score


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
        Class labels.

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


class GaussianNaiveBayes:
    """
    Gaussian Naive Bayes classifier implemented from scratch.

    Gaussian Naive Bayes assumes that the features are conditionally independent
    given the class label and that each feature follows a Gaussian distribution
    within each class.

    Parameters
    ----------
    var_smoothing : float, default=1e-9
        Portion of the largest variance added to each feature variance for
        numerical stability.

    Attributes
    ----------
    classes_ : np.ndarray
        Unique class labels.
    class_prior_ : np.ndarray
        Prior probability of each class.
    theta_ : np.ndarray
        Mean of each feature per class, with shape (n_classes, n_features).
    var_ : np.ndarray
        Variance of each feature per class, with shape (n_classes, n_features).
    epsilon_ : float
        Small value added to variances for numerical stability.
    n_features_in_ : int
        Number of features seen during fitting.
    """

    def __init__(self, var_smoothing: float = 1e-9) -> None:
        if var_smoothing < 0:
            raise ValueError("var_smoothing must be nonnegative.")

        self.var_smoothing = var_smoothing

        self.classes_: np.ndarray | None = None
        self.class_prior_: np.ndarray | None = None
        self.theta_: np.ndarray | None = None
        self.var_: np.ndarray | None = None
        self.epsilon_: float | None = None
        self.n_features_in_: int | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianNaiveBayes":
        """
        Fit the Gaussian Naive Bayes classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Class labels.

        Returns
        -------
        GaussianNaiveBayes
            Fitted classifier.
        """
        X, y = _validate_X_y(X, y)

        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]

        n_classes = len(self.classes_)
        n_features = X.shape[1]

        self.theta_ = np.zeros((n_classes, n_features), dtype=float)
        self.var_ = np.zeros((n_classes, n_features), dtype=float)
        self.class_prior_ = np.zeros(n_classes, dtype=float)

        max_variance = np.var(X, axis=0).max()
        self.epsilon_ = self.var_smoothing * max_variance

        for class_index, class_label in enumerate(self.classes_):
            X_class = X[y == class_label]

            self.theta_[class_index, :] = np.mean(X_class, axis=0)
            self.var_[class_index, :] = np.var(X_class, axis=0) + self.epsilon_
            self.class_prior_[class_index] = X_class.shape[0] / X.shape[0]

        return self

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict log-probabilities for each class.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features.

        Returns
        -------
        np.ndarray
            Log-probability matrix of shape (n_samples, n_classes).
        """
        self._check_is_fitted()
        X = _validate_X(X)
        self._check_n_features(X)

        log_proba = np.zeros((X.shape[0], len(self.classes_)), dtype=float)

        for class_index in range(len(self.classes_)):
            log_prior = np.log(self.class_prior_[class_index])
            log_likelihood = self._joint_log_likelihood(
                X,
                self.theta_[class_index],
                self.var_[class_index],
            )
            log_proba[:, class_index] = log_prior + log_likelihood

        return log_proba

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features.

        Returns
        -------
        np.ndarray
            Probability matrix of shape (n_samples, n_classes).
        """
        log_proba = self.predict_log_proba(X)

        shifted_log_proba = log_proba - np.max(log_proba, axis=1, keepdims=True)
        proba = np.exp(shifted_log_proba)

        return proba / np.sum(proba, axis=1, keepdims=True)

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
        log_proba = self.predict_log_proba(X)
        class_indices = np.argmax(log_proba, axis=1)

        return self.classes_[class_indices]

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

    @staticmethod
    def _joint_log_likelihood(
        X: np.ndarray,
        mean: np.ndarray,
        var: np.ndarray,
    ) -> np.ndarray:
        """
        Compute Gaussian log-likelihood for all samples for one class.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        mean : np.ndarray
            Feature means for one class.
        var : np.ndarray
            Feature variances for one class.

        Returns
        -------
        np.ndarray
            Log-likelihood values for each sample.
        """
        return -0.5 * np.sum(
            np.log(2.0 * np.pi * var) + ((X - mean) ** 2) / var,
            axis=1,
        )

    def _check_is_fitted(self) -> None:
        """Raise an error if the classifier has not been fitted."""
        if (
            self.classes_ is None
            or self.class_prior_ is None
            or self.theta_ is None
            or self.var_ is None
        ):
            raise ValueError("This GaussianNaiveBayes instance is not fitted yet.")

    def _check_n_features(self, X: np.ndarray) -> None:
        """Raise an error if the number of features does not match training data."""
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, but got {X.shape[1]}."
            )