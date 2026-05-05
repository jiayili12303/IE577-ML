"""
Gradient boosting regression implemented from scratch using NumPy.

This module includes:

- GradientBoostingRegressor: additive regression model using squared-error loss.

The implementation fits a sequence of shallow regression trees to the residuals
of the previous ensemble. It is intended for educational use and prioritizes
clarity over production-level optimization.
"""

from __future__ import annotations

import numpy as np

from jiayi_ml.metrics import r2_score


def _validate_X(X):
    """Convert input features to a 2D NumPy array of type float."""
    X = np.asarray(X, dtype=float)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    if X.ndim != 2:
        raise ValueError("X must be a 1D or 2D array.")

    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X must not be empty.")

    return X


def _validate_X_y(X, y):
    """Validate feature matrix X and target vector y."""
    X = _validate_X(X)
    y = np.asarray(y, dtype=float).ravel()

    if y.size == 0:
        raise ValueError("y must not be empty.")

    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must contain the same number of samples.")

    return X, y


class _RegressionTreeNode:
    """Internal regression tree node."""

    def __init__(
        self,
        value,
        feature_index=None,
        threshold=None,
        left=None,
        right=None,
    ):
        self.value = value
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right

    @property
    def is_leaf(self):
        return self.left is None and self.right is None


class _RegressionTree:
    """
    Simple CART-style regression tree for internal gradient boosting use.
    """

    def __init__(self, max_depth=3, min_samples_split=2, min_samples_leaf=1):
        if max_depth <= 0:
            raise ValueError("max_depth must be positive.")
        if min_samples_split <= 1:
            raise ValueError("min_samples_split must be greater than 1.")
        if min_samples_leaf <= 0:
            raise ValueError("min_samples_leaf must be positive.")

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.root_ = None
        self.n_features_in_ = None

    def fit(self, X, y):
        X, y = _validate_X_y(X, y)
        self.n_features_in_ = X.shape[1]
        self.root_ = self._build_tree(X, y, depth=0)
        return self

    def predict(self, X):
        if self.root_ is None:
            raise ValueError("This regression tree is not fitted yet.")

        X = _validate_X(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, but got {X.shape[1]}."
            )

        return np.array([self._predict_one(row, self.root_) for row in X])

    def _build_tree(self, X, y, depth):
        value = float(np.mean(y))

        if (
            depth >= self.max_depth
            or X.shape[0] < self.min_samples_split
            or np.allclose(y, y[0])
        ):
            return _RegressionTreeNode(value=value)

        split = self._best_split(X, y)

        if split is None:
            return _RegressionTreeNode(value=value)

        feature_index, threshold, left_mask, right_mask = split

        left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return _RegressionTreeNode(
            value=value,
            feature_index=feature_index,
            threshold=threshold,
            left=left,
            right=right,
        )

    def _best_split(self, X, y):
        best_feature = None
        best_threshold = None
        best_loss = np.inf
        best_left_mask = None
        best_right_mask = None

        current_loss = self._squared_error(y)

        for feature_index in range(X.shape[1]):
            values = np.unique(X[:, feature_index])

            if values.size <= 1:
                continue

            thresholds = (values[:-1] + values[1:]) / 2.0

            for threshold in thresholds:
                left_mask = X[:, feature_index] <= threshold
                right_mask = ~left_mask

                if (
                    np.sum(left_mask) < self.min_samples_leaf
                    or np.sum(right_mask) < self.min_samples_leaf
                ):
                    continue

                left_loss = self._squared_error(y[left_mask])
                right_loss = self._squared_error(y[right_mask])
                weighted_loss = left_loss + right_loss

                if weighted_loss < best_loss:
                    best_loss = weighted_loss
                    best_feature = feature_index
                    best_threshold = threshold
                    best_left_mask = left_mask
                    best_right_mask = right_mask

        if best_feature is None:
            return None

        if best_loss >= current_loss:
            return None

        return best_feature, best_threshold, best_left_mask, best_right_mask

    @staticmethod
    def _squared_error(y):
        if y.size == 0:
            return 0.0
        return float(np.sum((y - np.mean(y)) ** 2))

    def _predict_one(self, row, node):
        if node.is_leaf:
            return node.value

        if row[node.feature_index] <= node.threshold:
            return self._predict_one(row, node.left)

        return self._predict_one(row, node.right)


class GradientBoostingRegressor:
    """
    Gradient boosting regressor for squared-error loss.

    The model starts with a constant prediction equal to the mean of the target.
    It then sequentially fits shallow regression trees to the residuals and adds
    each tree's predictions to the ensemble.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of boosting stages.
    learning_rate : float, default=0.1
        Shrinkage applied to each tree's contribution.
    max_depth : int, default=3
        Maximum depth of each regression tree.
    min_samples_split : int, default=2
        Minimum number of samples required to split an internal tree node.
    min_samples_leaf : int, default=1
        Minimum number of samples required at each leaf node.

    Attributes
    ----------
    init_ : float
        Initial constant prediction.
    estimators_ : list
        List of fitted regression trees.
    train_loss_ : list[float]
        Training mean squared error after each boosting stage.
    n_features_in_ : int
        Number of features seen during fitting.
    """

    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=2,
        min_samples_leaf=1,
    ):
        if n_estimators <= 0:
            raise ValueError("n_estimators must be positive.")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if max_depth <= 0:
            raise ValueError("max_depth must be positive.")
        if min_samples_split <= 1:
            raise ValueError("min_samples_split must be greater than 1.")
        if min_samples_leaf <= 0:
            raise ValueError("min_samples_leaf must be positive.")

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

        self.init_ = None
        self.estimators_ = []
        self.train_loss_ = []
        self.n_features_in_ = None

    def fit(self, X, y):
        """
        Fit the gradient boosting regressor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Continuous target values.

        Returns
        -------
        GradientBoostingRegressor
            Fitted model.
        """
        X, y = _validate_X_y(X, y)

        self.n_features_in_ = X.shape[1]
        self.init_ = float(np.mean(y))
        self.estimators_ = []
        self.train_loss_ = []

        y_pred = np.full(y.shape, self.init_, dtype=float)

        for _ in range(self.n_estimators):
            residuals = y - y_pred

            tree = _RegressionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
            )
            tree.fit(X, residuals)

            update = tree.predict(X)
            y_pred += self.learning_rate * update

            self.estimators_.append(tree)
            self.train_loss_.append(float(np.mean((y - y_pred) ** 2)))

        return self

    def predict(self, X):
        """
        Predict continuous target values.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features.

        Returns
        -------
        np.ndarray
            Predicted values.
        """
        self._check_is_fitted()
        X = _validate_X(X)
        self._check_n_features(X)

        y_pred = np.full(X.shape[0], self.init_, dtype=float)

        for tree in self.estimators_:
            y_pred += self.learning_rate * tree.predict(X)

        return y_pred

    def staged_predict(self, X):
        """
        Generate predictions after each boosting stage.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features.

        Yields
        ------
        np.ndarray
            Predictions after each boosting stage.
        """
        self._check_is_fitted()
        X = _validate_X(X)
        self._check_n_features(X)

        y_pred = np.full(X.shape[0], self.init_, dtype=float)

        for tree in self.estimators_:
            y_pred = y_pred + self.learning_rate * tree.predict(X)
            yield y_pred.copy()

    def score(self, X, y):
        """
        Compute R-squared score.

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

    def _check_is_fitted(self):
        if self.init_ is None or len(self.estimators_) == 0:
            raise ValueError("This GradientBoostingRegressor instance is not fitted yet.")

    def _check_n_features(self, X):
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, but got {X.shape[1]}."
            )