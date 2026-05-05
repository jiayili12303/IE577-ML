"""
Linear support vector machine classifier implemented from scratch using NumPy.

This module includes:

- LinearSVM: binary linear support vector machine classifier.

The implementation optimizes the primal soft-margin SVM objective using
batch gradient descent with hinge loss and L2 regularization.
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


class LinearSVM:
    """
    Binary linear support vector machine classifier.

    This classifier minimizes a soft-margin SVM objective using hinge loss and
    L2 regularization. The model learns a linear decision boundary.

    Parameters
    ----------
    learning_rate : float, default=0.01
        Step size for gradient descent.
    max_iter : int, default=1000
        Maximum number of gradient descent iterations.
    alpha : float, default=0.01
        L2 regularization strength. Larger values increase weight shrinkage.
    fit_intercept : bool, default=True
        Whether to estimate an intercept term.
    tol : float, default=1e-6
        Convergence tolerance based on the absolute change in objective value.

    Attributes
    ----------
    coef_ : np.ndarray
        Learned coefficient vector of shape (n_features,).
    intercept_ : float
        Learned intercept term.
    classes_ : np.ndarray
        Original class labels in sorted order.
    n_features_in_ : int
        Number of features seen during fitting.
    n_iter_ : int
        Number of iterations completed during fitting.
    losses_ : list[float]
        Objective value at each iteration.
    """

    def __init__(
        self,
        learning_rate=0.01,
        max_iter=1000,
        alpha=0.01,
        fit_intercept=True,
        tol=1e-6,
    ):
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")

        if max_iter <= 0:
            raise ValueError("max_iter must be positive.")

        if alpha < 0:
            raise ValueError("alpha must be nonnegative.")

        if tol < 0:
            raise ValueError("tol must be nonnegative.")

        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.tol = tol

        self.coef_ = None
        self.intercept_ = None
        self.classes_ = None
        self.n_features_in_ = None
        self.n_iter_ = None
        self.losses_ = []

    def fit(self, X, y):
        """
        Fit the linear SVM classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Binary class labels.

        Returns
        -------
        LinearSVM
            Fitted classifier.

        Raises
        ------
        ValueError
            If y does not contain exactly two classes.
        """
        X, y = _validate_X_y(X, y)

        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError("LinearSVM supports binary classification only.")

        self.classes_ = classes
        self.n_features_in_ = X.shape[1]

        y_binary = np.where(y == classes[1], 1.0, -1.0)

        coef = np.zeros(X.shape[1], dtype=float)
        intercept = 0.0
        self.losses_ = []

        previous_loss = None

        for iteration in range(1, self.max_iter + 1):
            scores = X @ coef + intercept
            margins = y_binary * scores
            active = margins < 1.0

            if np.any(active):
                gradient_coef = (
                    self.alpha * coef
                    - np.mean(y_binary[active, None] * X[active], axis=0)
                )
                gradient_intercept = -float(np.mean(y_binary[active]))
            else:
                gradient_coef = self.alpha * coef
                gradient_intercept = 0.0

            coef -= self.learning_rate * gradient_coef

            if self.fit_intercept:
                intercept -= self.learning_rate * gradient_intercept

            loss = self._objective(X, y_binary, coef, intercept)
            self.losses_.append(loss)

            if previous_loss is not None and abs(previous_loss - loss) < self.tol:
                break

            previous_loss = loss

        self.coef_ = coef
        self.intercept_ = float(intercept if self.fit_intercept else 0.0)
        self.n_iter_ = iteration

        return self

    def decision_function(self, X):
        """
        Compute signed linear decision scores.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features.

        Returns
        -------
        np.ndarray
            Linear decision scores. Positive values correspond to classes_[1].
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
        return np.where(scores >= 0.0, self.classes_[1], self.classes_[0])

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

    def _objective(self, X, y_binary, coef, intercept):
        """
        Compute the soft-margin linear SVM objective.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        y_binary : np.ndarray
            Labels encoded as -1 and 1.
        coef : np.ndarray
            Coefficient vector.
        intercept : float
            Intercept term.

        Returns
        -------
        float
            Objective value.
        """
        margins = y_binary * (X @ coef + intercept)
        hinge_loss = np.mean(np.maximum(0.0, 1.0 - margins))
        regularization = 0.5 * self.alpha * np.sum(coef ** 2)

        return float(hinge_loss + regularization)

    def _check_is_fitted(self):
        """Raise an error if the model has not been fitted."""
        if self.coef_ is None or self.intercept_ is None or self.classes_ is None:
            raise ValueError("This LinearSVM instance is not fitted yet.")

    def _check_n_features(self, X):
        """Raise an error if the number of features does not match training data."""
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, but got {X.shape[1]}."
            )