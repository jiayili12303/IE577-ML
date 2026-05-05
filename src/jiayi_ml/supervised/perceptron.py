"""
Perceptron classifier implemented from scratch using NumPy.

The perceptron is a linear binary classifier and one of the earliest neural
network models. It updates its weights whenever a training sample is
misclassified.
"""

from __future__ import annotations

import numpy as np

from jiayi_ml.metrics import accuracy_score


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


def _validate_X_y(X, y):
    """
    Validate feature matrix X and target vector y.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    y : array-like of shape (n_samples,)
        Binary class labels.

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


class Perceptron:
    """
    Binary perceptron classifier implemented from scratch.

    Parameters
    ----------
    learning_rate : float, default=0.01
        Step size for weight updates.
    max_iter : int, default=1000
        Maximum number of passes over the training data.
    fit_intercept : bool, default=True
        Whether to estimate an intercept term.
    shuffle : bool, default=True
        Whether to shuffle training samples at each epoch.
    random_state : int or None, default=None
        Random seed used when shuffle=True.

    Attributes
    ----------
    coef_ : np.ndarray
        Learned coefficient vector of shape (n_features,).
    intercept_ : float
        Learned intercept.
    classes_ : np.ndarray
        Original class labels in sorted order.
    n_features_in_ : int
        Number of features seen during fitting.
    n_iter_ : int
        Number of epochs completed.
    errors_ : list[int]
        Number of misclassified samples at each epoch.
    """

    def __init__(
        self,
        learning_rate=0.01,
        max_iter=1000,
        fit_intercept=True,
        shuffle=True,
        random_state=None,
    ):
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")

        if max_iter <= 0:
            raise ValueError("max_iter must be positive.")

        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.fit_intercept = fit_intercept
        self.shuffle = shuffle
        self.random_state = random_state

        self.coef_ = None
        self.intercept_ = None
        self.classes_ = None
        self.n_features_in_ = None
        self.n_iter_ = None
        self.errors_ = []

    def fit(self, X, y):
        """
        Fit the perceptron classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Binary class labels.

        Returns
        -------
        Perceptron
            Fitted classifier.

        Raises
        ------
        ValueError
            If y does not contain exactly two classes.
        """
        X, y = _validate_X_y(X, y)

        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError("Perceptron supports binary classification only.")

        self.classes_ = classes
        self.n_features_in_ = X.shape[1]

        y_binary = np.where(y == classes[1], 1, -1)

        rng = np.random.default_rng(self.random_state)

        coef = np.zeros(X.shape[1], dtype=float)
        intercept = 0.0
        self.errors_ = []

        for epoch in range(1, self.max_iter + 1):
            errors = 0
            indices = np.arange(X.shape[0])

            if self.shuffle:
                rng.shuffle(indices)

            for idx in indices:
                linear_output = X[idx] @ coef + intercept
                predicted = 1 if linear_output >= 0 else -1

                if predicted != y_binary[idx]:
                    update = self.learning_rate * y_binary[idx]
                    coef += update * X[idx]

                    if self.fit_intercept:
                        intercept += update

                    errors += 1

            self.errors_.append(errors)

            if errors == 0:
                break

        self.coef_ = coef
        self.intercept_ = float(intercept if self.fit_intercept else 0.0)
        self.n_iter_ = epoch

        return self

    def decision_function(self, X):
        """
        Compute signed distance-like linear scores.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features.

        Returns
        -------
        np.ndarray
            Linear scores. Positive values correspond to classes_[1].
        """
        self._check_is_fitted()
        X = _validate_X(X)
        self._check_n_features(X)

        return X @ self.coef_ + self.intercept_

    def predict(self, X):
        """
        Predict class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features.

        Returns
        -------
        np.ndarray
            Predicted class labels.
        """
        scores = self.decision_function(X)
        positive_mask = scores >= 0

        return np.where(positive_mask, self.classes_[1], self.classes_[0])

    def score(self, X, y):
        """
        Compute classification accuracy.

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

    def _check_is_fitted(self):
        """Raise an error if the model has not been fitted."""
        if self.coef_ is None or self.intercept_ is None or self.classes_ is None:
            raise ValueError("This Perceptron instance is not fitted yet.")

    def _check_n_features(self, X):
        """Raise an error if the number of features does not match training data."""
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, but got {X.shape[1]}."
            )