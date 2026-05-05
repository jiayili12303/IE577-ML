"""
Logistic regression implemented from scratch using NumPy.

This module provides a binary LogisticRegression classifier with a
scikit-learn-like API. The model is trained using gradient descent and supports
optional L1 or L2 regularization.
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

    Raises
    ------
    ValueError
        If X is empty or has more than two dimensions.
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
    Validate feature matrix X and binary target vector y.

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

    Raises
    ------
    ValueError
        If X and y have incompatible shapes or are empty.
    """
    X = _validate_X(X)
    y = np.asarray(y).ravel()

    if y.size == 0:
        raise ValueError("y must not be empty.")

    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must contain the same number of samples.")

    return X, y


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Compute the sigmoid function in a numerically stable way.

    Parameters
    ----------
    z : np.ndarray
        Linear predictor.

    Returns
    -------
    np.ndarray
        Sigmoid-transformed values.
    """
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


class LogisticRegression:
    """
    Binary logistic regression classifier implemented from scratch.

    Parameters
    ----------
    learning_rate : float, default=0.01
        Step size used in gradient descent.
    max_iter : int, default=1000
        Maximum number of gradient descent iterations.
    fit_intercept : bool, default=True
        Whether to estimate an intercept term.
    penalty : {None, "l1", "l2"}, default=None
        Type of regularization to apply to the coefficients.
    alpha : float, default=0.0
        Regularization strength. The intercept is not regularized.
    tol : float, default=1e-6
        Tolerance for convergence based on the change in loss.

    Attributes
    ----------
    coef_ : np.ndarray
        Estimated coefficients of shape (n_features,).
    intercept_ : float
        Estimated intercept.
    classes_ : np.ndarray
        Original class labels in sorted order.
    losses_ : list[float]
        Binary cross-entropy loss values recorded during training.
    n_features_in_ : int
        Number of features seen during fitting.
    n_iter_ : int
        Number of iterations completed.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        max_iter: int = 1000,
        fit_intercept: bool = True,
        penalty: str | None = None,
        alpha: float = 0.0,
        tol: float = 1e-6,
    ) -> None:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")

        if max_iter <= 0:
            raise ValueError("max_iter must be positive.")

        if penalty not in {None, "l1", "l2"}:
            raise ValueError("penalty must be one of None, 'l1', or 'l2'.")

        if alpha < 0:
            raise ValueError("alpha must be nonnegative.")

        if tol <= 0:
            raise ValueError("tol must be positive.")

        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.fit_intercept = fit_intercept
        self.penalty = penalty
        self.alpha = alpha
        self.tol = tol

        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None
        self.classes_: np.ndarray | None = None
        self.losses_: list[float] = []
        self.n_features_in_: int | None = None
        self.n_iter_: int | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        """
        Fit the logistic regression model using gradient descent.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Binary class labels.

        Returns
        -------
        LogisticRegression
            Fitted model.

        Raises
        ------
        ValueError
            If y does not contain exactly two classes.
        """
        X, y = _validate_X_y(X, y)
        n_samples, n_features = X.shape

        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError("LogisticRegression supports binary classification only.")

        self.classes_ = classes
        self.n_features_in_ = n_features

        y_binary = (y == classes[1]).astype(float)

        coef = np.zeros(n_features)
        intercept = 0.0
        previous_loss = np.inf
        self.losses_ = []

        for iteration in range(1, self.max_iter + 1):
            linear_output = X @ coef + intercept
            probabilities = _sigmoid(linear_output)

            error = probabilities - y_binary

            gradient_coef = (X.T @ error) / n_samples
            gradient_intercept = float(np.mean(error))

            if self.penalty == "l2":
                gradient_coef += self.alpha * coef
            elif self.penalty == "l1":
                gradient_coef += self.alpha * np.sign(coef)

            coef -= self.learning_rate * gradient_coef

            if self.fit_intercept:
                intercept -= self.learning_rate * gradient_intercept
            else:
                intercept = 0.0

            loss = self._binary_cross_entropy(y_binary, probabilities, coef)
            self.losses_.append(loss)

            if abs(previous_loss - loss) < self.tol:
                break

            previous_loss = loss

        self.coef_ = coef
        self.intercept_ = float(intercept)
        self.n_iter_ = iteration

        return self

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
            Probability matrix of shape (n_samples, 2). The first column
            corresponds to classes_[0], and the second column corresponds to
            classes_[1].
        """
        self._check_is_fitted()
        X = _validate_X(X)
        self._check_n_features(X)

        positive_prob = _sigmoid(X @ self.coef_ + self.intercept_)
        negative_prob = 1.0 - positive_prob

        return np.column_stack([negative_prob, positive_prob])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features.
        threshold : float, default=0.5
            Probability threshold for assigning the positive class.

        Returns
        -------
        np.ndarray
            Predicted class labels.

        Raises
        ------
        ValueError
            If threshold is not between 0 and 1.
        """
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1.")

        probabilities = self.predict_proba(X)[:, 1]
        predicted_binary = probabilities >= threshold

        return np.where(predicted_binary, self.classes_[1], self.classes_[0])

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

    def _binary_cross_entropy(
        self,
        y_true: np.ndarray,
        probabilities: np.ndarray,
        coef: np.ndarray,
    ) -> float:
        """
        Compute binary cross-entropy loss with optional regularization.

        Parameters
        ----------
        y_true : np.ndarray
            Binary target values encoded as 0 and 1.
        probabilities : np.ndarray
            Predicted positive-class probabilities.
        coef : np.ndarray
            Current coefficient vector.

        Returns
        -------
        float
            Binary cross-entropy loss.
        """
        eps = 1e-15
        probabilities = np.clip(probabilities, eps, 1.0 - eps)

        loss = -np.mean(
            y_true * np.log(probabilities)
            + (1.0 - y_true) * np.log(1.0 - probabilities)
        )

        if self.penalty == "l2":
            loss += 0.5 * self.alpha * float(np.sum(coef ** 2))
        elif self.penalty == "l1":
            loss += self.alpha * float(np.sum(np.abs(coef)))

        return float(loss)

    def _check_is_fitted(self) -> None:
        """Raise an error if the model has not been fitted."""
        if self.coef_ is None or self.intercept_ is None or self.classes_ is None:
            raise ValueError("This LogisticRegression instance is not fitted yet.")

    def _check_n_features(self, X: np.ndarray) -> None:
        """Raise an error if the number of features does not match training data."""
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, but got {X.shape[1]}."
            )