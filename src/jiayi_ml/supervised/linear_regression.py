"""
Linear regression models implemented from scratch using NumPy.

This module includes:

- LinearRegression: ordinary least squares regression.
- RidgeRegression: linear regression with L2 regularization.
- LassoRegression: linear regression with L1 regularization using coordinate descent.

All classes follow a scikit-learn-like API with fit, predict, and score methods.
"""

from __future__ import annotations

import numpy as np

from jiayi_ml.metrics import r2_score


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
    Validate feature matrix X and target vector y.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    y : array-like of shape (n_samples,)
        Target values.

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
    y = np.asarray(y, dtype=float).ravel()

    if y.size == 0:
        raise ValueError("y must not be empty.")

    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must contain the same number of samples.")

    return X, y


def _soft_threshold(value: float, threshold: float) -> float:
    """
    Apply the soft-thresholding operator used in Lasso coordinate descent.

    Parameters
    ----------
    value : float
        Input value.
    threshold : float
        Nonnegative threshold value.

    Returns
    -------
    float
        Thresholded value.
    """
    if value > threshold:
        return value - threshold

    if value < -threshold:
        return value + threshold

    return 0.0


class LinearRegression:
    """
    Ordinary least squares linear regression implemented from scratch.

    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to estimate an intercept term.

    Attributes
    ----------
    coef_ : np.ndarray
        Estimated coefficients of shape (n_features,).
    intercept_ : float
        Estimated intercept.
    n_features_in_ : int
        Number of features seen during fitting.
    """

    def __init__(self, fit_intercept: bool = True) -> None:
        self.fit_intercept = fit_intercept
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None
        self.n_features_in_: int | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        """
        Fit the linear regression model using the Moore-Penrose pseudoinverse.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        LinearRegression
            Fitted model.
        """
        X, y = _validate_X_y(X, y)
        self.n_features_in_ = X.shape[1]

        if self.fit_intercept:
            X_design = np.column_stack([np.ones(X.shape[0]), X])
            params = np.linalg.pinv(X_design) @ y
            self.intercept_ = float(params[0])
            self.coef_ = params[1:]
        else:
            self.coef_ = np.linalg.pinv(X) @ y
            self.intercept_ = 0.0

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

        return X @ self.coef_ + self.intercept_

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

    def _check_is_fitted(self) -> None:
        """Raise an error if the model has not been fitted."""
        if self.coef_ is None or self.intercept_ is None:
            raise ValueError("This LinearRegression instance is not fitted yet.")

    def _check_n_features(self, X: np.ndarray) -> None:
        """Raise an error if the number of features does not match training data."""
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, but got {X.shape[1]}."
            )


class RidgeRegression:
    """
    Ridge regression implemented from scratch using the closed-form solution.

    Ridge regression minimizes the residual sum of squares with an L2 penalty
    on the coefficients. The intercept is not regularized.

    Parameters
    ----------
    alpha : float, default=1.0
        L2 regularization strength. Must be nonnegative.
    fit_intercept : bool, default=True
        Whether to estimate an intercept term.

    Attributes
    ----------
    coef_ : np.ndarray
        Estimated coefficients of shape (n_features,).
    intercept_ : float
        Estimated intercept.
    n_features_in_ : int
        Number of features seen during fitting.
    """

    def __init__(self, alpha: float = 1.0, fit_intercept: bool = True) -> None:
        if alpha < 0:
            raise ValueError("alpha must be nonnegative.")

        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None
        self.n_features_in_: int | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeRegression":
        """
        Fit the ridge regression model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        RidgeRegression
            Fitted model.
        """
        X, y = _validate_X_y(X, y)
        self.n_features_in_ = X.shape[1]

        if self.fit_intercept:
            X_design = np.column_stack([np.ones(X.shape[0]), X])
            penalty = np.eye(X_design.shape[1])
            penalty[0, 0] = 0.0
        else:
            X_design = X
            penalty = np.eye(X.shape[1])

        gram_matrix = X_design.T @ X_design
        params = np.linalg.pinv(gram_matrix + self.alpha * penalty) @ X_design.T @ y

        if self.fit_intercept:
            self.intercept_ = float(params[0])
            self.coef_ = params[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = params

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

        return X @ self.coef_ + self.intercept_

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

    def _check_is_fitted(self) -> None:
        """Raise an error if the model has not been fitted."""
        if self.coef_ is None or self.intercept_ is None:
            raise ValueError("This RidgeRegression instance is not fitted yet.")

    def _check_n_features(self, X: np.ndarray) -> None:
        """Raise an error if the number of features does not match training data."""
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, but got {X.shape[1]}."
            )


class LassoRegression:
    """
    Lasso regression implemented from scratch using coordinate descent.

    Lasso regression minimizes squared error with an L1 penalty on the
    coefficients. The intercept is not regularized.

    Parameters
    ----------
    alpha : float, default=1.0
        L1 regularization strength. Must be nonnegative.
    fit_intercept : bool, default=True
        Whether to estimate an intercept term.
    max_iter : int, default=1000
        Maximum number of coordinate descent iterations.
    tol : float, default=1e-4
        Tolerance for convergence.

    Attributes
    ----------
    coef_ : np.ndarray
        Estimated coefficients of shape (n_features,).
    intercept_ : float
        Estimated intercept.
    n_features_in_ : int
        Number of features seen during fitting.
    n_iter_ : int
        Number of coordinate descent iterations completed.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-4,
    ) -> None:
        if alpha < 0:
            raise ValueError("alpha must be nonnegative.")

        if max_iter <= 0:
            raise ValueError("max_iter must be positive.")

        if tol <= 0:
            raise ValueError("tol must be positive.")

        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol

        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None
        self.n_features_in_: int | None = None
        self.n_iter_: int | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LassoRegression":
        """
        Fit the lasso regression model using coordinate descent.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        LassoRegression
            Fitted model.
        """
        X, y = _validate_X_y(X, y)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        if self.fit_intercept:
            X_offset = np.mean(X, axis=0)
            y_offset = np.mean(y)
            X_work = X - X_offset
            y_work = y - y_offset
        else:
            X_offset = np.zeros(n_features)
            y_offset = 0.0
            X_work = X.copy()
            y_work = y.copy()

        coef = np.zeros(n_features)
        column_squared_norms = np.sum(X_work ** 2, axis=0)

        for iteration in range(1, self.max_iter + 1):
            coef_old = coef.copy()

            for feature_idx in range(n_features):
                if column_squared_norms[feature_idx] == 0:
                    coef[feature_idx] = 0.0
                    continue

                residual = (
                    y_work
                    - X_work @ coef
                    + coef[feature_idx] * X_work[:, feature_idx]
                )

                rho = float(X_work[:, feature_idx] @ residual)
                threshold = self.alpha * n_samples

                coef[feature_idx] = _soft_threshold(rho, threshold) / column_squared_norms[
                    feature_idx
                ]

            coef_change = np.linalg.norm(coef - coef_old)
            coef_scale = np.linalg.norm(coef_old) + self.tol

            if coef_change <= self.tol * coef_scale:
                break

        self.coef_ = coef
        self.intercept_ = float(y_offset - X_offset @ coef)
        self.n_iter_ = iteration

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

        return X @ self.coef_ + self.intercept_

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

    def _check_is_fitted(self) -> None:
        """Raise an error if the model has not been fitted."""
        if self.coef_ is None or self.intercept_ is None:
            raise ValueError("This LassoRegression instance is not fitted yet.")

    def _check_n_features(self, X: np.ndarray) -> None:
        """Raise an error if the number of features does not match training data."""
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, but got {X.shape[1]}."
            )