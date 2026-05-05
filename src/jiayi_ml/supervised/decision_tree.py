"""
Decision tree models implemented from scratch using NumPy.

This module includes:

- DecisionTreeClassifier: CART-style decision tree for classification.
- DecisionTreeRegressor: CART-style decision tree for regression.

The implementation supports recursive binary splitting, maximum depth control,
minimum samples per split/leaf, optional random feature selection, and simple
feature importance scores based on impurity reduction.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter

import numpy as np

from jiayi_ml.metrics import accuracy_score, r2_score


@dataclass
class _TreeNode:
    """Internal tree node used by decision tree models."""

    prediction: object
    impurity: float
    n_samples: int
    feature_index: int | None = None
    threshold: float | None = None
    left: "_TreeNode | None" = None
    right: "_TreeNode | None" = None

    @property
    def is_leaf(self) -> bool:
        """Return True if the node is a leaf node."""
        return self.left is None and self.right is None


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


class DecisionTreeClassifier:
    """
    Decision tree classifier implemented from scratch.

    Parameters
    ----------
    max_depth : int or None, default=None
        Maximum depth of the tree. If None, nodes are expanded until pure or
        until stopping criteria are met.
    min_samples_split : int, default=2
        Minimum number of samples required to split an internal node.
    min_samples_leaf : int, default=1
        Minimum number of samples required at each leaf node.
    criterion : {"gini", "entropy"}, default="gini"
        Impurity criterion used for splitting.
    max_features : int, {"sqrt", "log2"}, or None, default=None
        Number of features considered at each split. If None, all features are
        considered.
    random_state : int or None, default=None
        Random seed used when max_features selects a random feature subset.

    Attributes
    ----------
    tree_ : _TreeNode
        Root node of the fitted tree.
    classes_ : np.ndarray
        Unique class labels.
    feature_importances_ : np.ndarray
        Normalized feature importance scores.
    n_features_in_ : int
        Number of features seen during fitting.
    """

    def __init__(
        self,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        criterion="gini",
        max_features=None,
        random_state=None,
    ):
        if max_depth is not None and max_depth <= 0:
            raise ValueError("max_depth must be positive or None.")

        if min_samples_split < 2:
            raise ValueError("min_samples_split must be at least 2.")

        if min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be at least 1.")

        if criterion not in {"gini", "entropy"}:
            raise ValueError("criterion must be either 'gini' or 'entropy'.")

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.max_features = max_features
        self.random_state = random_state

        self.tree_ = None
        self.classes_ = None
        self.feature_importances_ = None
        self.n_features_in_ = None
        self._rng = None
        self._raw_feature_importances = None

    def fit(self, X, y):
        """
        Fit the decision tree classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Class labels.

        Returns
        -------
        DecisionTreeClassifier
            Fitted classifier.
        """
        X, y = _validate_X_y(X, y)

        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        self._rng = np.random.default_rng(self.random_state)
        self._raw_feature_importances = np.zeros(self.n_features_in_, dtype=float)

        self.tree_ = self._build_tree(X, y, depth=0)

        total_importance = np.sum(self._raw_feature_importances)
        if total_importance > 0:
            self.feature_importances_ = self._raw_feature_importances / total_importance
        else:
            self.feature_importances_ = self._raw_feature_importances

        return self

    def predict(self, X):
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

        return np.array([self._predict_one(row, self.tree_) for row in X])

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

    def _build_tree(self, X, y, depth):
        prediction = self._majority_class(y)
        impurity = self._impurity(y)
        node = _TreeNode(prediction=prediction, impurity=impurity, n_samples=len(y))

        if self._should_stop(y, depth):
            return node

        split = self._best_split(X, y)
        if split is None:
            return node

        feature_index, threshold, left_mask, right_mask, impurity_decrease = split

        self._raw_feature_importances[feature_index] += impurity_decrease * len(y)

        node.feature_index = feature_index
        node.threshold = threshold
        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return node

    def _should_stop(self, y, depth):
        if len(np.unique(y)) == 1:
            return True

        if self.max_depth is not None and depth >= self.max_depth:
            return True

        if len(y) < self.min_samples_split:
            return True

        return False

    def _best_split(self, X, y):
        n_samples, n_features = X.shape
        parent_impurity = self._impurity(y)

        best_feature = None
        best_threshold = None
        best_left_mask = None
        best_right_mask = None
        best_impurity_decrease = 0.0

        feature_indices = self._feature_indices(n_features)

        for feature_index in feature_indices:
            values = np.unique(X[:, feature_index])

            if values.size <= 1:
                continue

            thresholds = (values[:-1] + values[1:]) / 2.0

            for threshold in thresholds:
                left_mask = X[:, feature_index] <= threshold
                right_mask = ~left_mask

                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)

                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue

                left_impurity = self._impurity(y[left_mask])
                right_impurity = self._impurity(y[right_mask])

                weighted_impurity = (
                    n_left / n_samples * left_impurity
                    + n_right / n_samples * right_impurity
                )

                impurity_decrease = parent_impurity - weighted_impurity

                if impurity_decrease > best_impurity_decrease:
                    best_feature = feature_index
                    best_threshold = threshold
                    best_left_mask = left_mask
                    best_right_mask = right_mask
                    best_impurity_decrease = impurity_decrease

        if best_feature is None:
            return None

        return (
            best_feature,
            best_threshold,
            best_left_mask,
            best_right_mask,
            best_impurity_decrease,
        )

    def _impurity(self, y):
        counts = np.array(list(Counter(y).values()), dtype=float)
        probabilities = counts / np.sum(counts)

        if self.criterion == "gini":
            return float(1.0 - np.sum(probabilities ** 2))

        probabilities = probabilities[probabilities > 0]
        return float(-np.sum(probabilities * np.log2(probabilities)))

    @staticmethod
    def _majority_class(y):
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]

    def _feature_indices(self, n_features):
        if self.max_features is None:
            return np.arange(n_features)

        if self.max_features == "sqrt":
            n_selected = max(1, int(np.sqrt(n_features)))
        elif self.max_features == "log2":
            n_selected = max(1, int(np.log2(n_features)))
        elif isinstance(self.max_features, int):
            if self.max_features <= 0 or self.max_features > n_features:
                raise ValueError("max_features must be between 1 and n_features.")
            n_selected = self.max_features
        else:
            raise ValueError("max_features must be None, an int, 'sqrt', or 'log2'.")

        return self._rng.choice(n_features, size=n_selected, replace=False)

    def _predict_one(self, row, node):
        while not node.is_leaf:
            if row[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right

        return node.prediction

    def _check_is_fitted(self):
        if self.tree_ is None:
            raise ValueError("This DecisionTreeClassifier instance is not fitted yet.")

    def _check_n_features(self, X):
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, but got {X.shape[1]}."
            )


class DecisionTreeRegressor:
    """
    Decision tree regressor implemented from scratch.

    Parameters
    ----------
    max_depth : int or None, default=None
        Maximum depth of the tree.
    min_samples_split : int, default=2
        Minimum number of samples required to split an internal node.
    min_samples_leaf : int, default=1
        Minimum number of samples required at each leaf node.
    criterion : {"squared_error", "absolute_error"}, default="squared_error"
        Impurity criterion used for splitting.
    max_features : int, {"sqrt", "log2"}, or None, default=None
        Number of features considered at each split.
    random_state : int or None, default=None
        Random seed used when max_features selects a random feature subset.

    Attributes
    ----------
    tree_ : _TreeNode
        Root node of the fitted tree.
    feature_importances_ : np.ndarray
        Normalized feature importance scores.
    n_features_in_ : int
        Number of features seen during fitting.
    """

    def __init__(
        self,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        criterion="squared_error",
        max_features=None,
        random_state=None,
    ):
        if max_depth is not None and max_depth <= 0:
            raise ValueError("max_depth must be positive or None.")

        if min_samples_split < 2:
            raise ValueError("min_samples_split must be at least 2.")

        if min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be at least 1.")

        if criterion not in {"squared_error", "absolute_error"}:
            raise ValueError(
                "criterion must be either 'squared_error' or 'absolute_error'."
            )

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.max_features = max_features
        self.random_state = random_state

        self.tree_ = None
        self.feature_importances_ = None
        self.n_features_in_ = None
        self._rng = None
        self._raw_feature_importances = None

    def fit(self, X, y):
        """
        Fit the decision tree regressor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        DecisionTreeRegressor
            Fitted regressor.
        """
        X, y = _validate_X_y(X, y)
        y = y.astype(float)

        self.n_features_in_ = X.shape[1]
        self._rng = np.random.default_rng(self.random_state)
        self._raw_feature_importances = np.zeros(self.n_features_in_, dtype=float)

        self.tree_ = self._build_tree(X, y, depth=0)

        total_importance = np.sum(self._raw_feature_importances)
        if total_importance > 0:
            self.feature_importances_ = self._raw_feature_importances / total_importance
        else:
            self.feature_importances_ = self._raw_feature_importances

        return self

    def predict(self, X):
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

        return np.array([self._predict_one(row, self.tree_) for row in X], dtype=float)

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

    def _build_tree(self, X, y, depth):
        prediction = float(np.mean(y))
        impurity = self._impurity(y)
        node = _TreeNode(prediction=prediction, impurity=impurity, n_samples=len(y))

        if self._should_stop(y, depth):
            return node

        split = self._best_split(X, y)
        if split is None:
            return node

        feature_index, threshold, left_mask, right_mask, impurity_decrease = split

        self._raw_feature_importances[feature_index] += impurity_decrease * len(y)

        node.feature_index = feature_index
        node.threshold = threshold
        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return node

    def _should_stop(self, y, depth):
        if np.all(y == y[0]):
            return True

        if self.max_depth is not None and depth >= self.max_depth:
            return True

        if len(y) < self.min_samples_split:
            return True

        return False

    def _best_split(self, X, y):
        n_samples, n_features = X.shape
        parent_impurity = self._impurity(y)

        best_feature = None
        best_threshold = None
        best_left_mask = None
        best_right_mask = None
        best_impurity_decrease = 0.0

        feature_indices = self._feature_indices(n_features)

        for feature_index in feature_indices:
            values = np.unique(X[:, feature_index])

            if values.size <= 1:
                continue

            thresholds = (values[:-1] + values[1:]) / 2.0

            for threshold in thresholds:
                left_mask = X[:, feature_index] <= threshold
                right_mask = ~left_mask

                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)

                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue

                left_impurity = self._impurity(y[left_mask])
                right_impurity = self._impurity(y[right_mask])

                weighted_impurity = (
                    n_left / n_samples * left_impurity
                    + n_right / n_samples * right_impurity
                )

                impurity_decrease = parent_impurity - weighted_impurity

                if impurity_decrease > best_impurity_decrease:
                    best_feature = feature_index
                    best_threshold = threshold
                    best_left_mask = left_mask
                    best_right_mask = right_mask
                    best_impurity_decrease = impurity_decrease

        if best_feature is None:
            return None

        return (
            best_feature,
            best_threshold,
            best_left_mask,
            best_right_mask,
            best_impurity_decrease,
        )

    def _impurity(self, y):
        if self.criterion == "squared_error":
            return float(np.mean((y - np.mean(y)) ** 2))

        return float(np.mean(np.abs(y - np.median(y))))

    def _feature_indices(self, n_features):
        if self.max_features is None:
            return np.arange(n_features)

        if self.max_features == "sqrt":
            n_selected = max(1, int(np.sqrt(n_features)))
        elif self.max_features == "log2":
            n_selected = max(1, int(np.log2(n_features)))
        elif isinstance(self.max_features, int):
            if self.max_features <= 0 or self.max_features > n_features:
                raise ValueError("max_features must be between 1 and n_features.")
            n_selected = self.max_features
        else:
            raise ValueError("max_features must be None, an int, 'sqrt', or 'log2'.")

        return self._rng.choice(n_features, size=n_selected, replace=False)

    def _predict_one(self, row, node):
        while not node.is_leaf:
            if row[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right

        return node.prediction

    def _check_is_fitted(self):
        if self.tree_ is None:
            raise ValueError("This DecisionTreeRegressor instance is not fitted yet.")

    def _check_n_features(self, X):
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, but got {X.shape[1]}."
            )