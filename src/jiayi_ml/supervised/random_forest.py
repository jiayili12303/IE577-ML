"""
Random forest models implemented from scratch using NumPy.

This module includes:

- RandomForestClassifier: ensemble of decision tree classifiers.
- RandomForestRegressor: ensemble of decision tree regressors.

Each tree is trained on a bootstrap sample of the training data. Random feature
selection is passed to the underlying decision trees through max_features.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from jiayi_ml.metrics import accuracy_score, r2_score
from jiayi_ml.supervised.decision_tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
)


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


class RandomForestClassifier:
    """
    Random forest classifier implemented from scratch.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of decision trees in the forest.
    max_depth : int or None, default=None
        Maximum depth of each tree.
    min_samples_split : int, default=2
        Minimum number of samples required to split an internal node.
    min_samples_leaf : int, default=1
        Minimum number of samples required at each leaf node.
    criterion : {"gini", "entropy"}, default="gini"
        Impurity criterion used by each decision tree.
    max_features : int, {"sqrt", "log2"}, or None, default="sqrt"
        Number of features considered at each split.
    bootstrap : bool, default=True
        Whether to train each tree on a bootstrap sample.
    random_state : int or None, default=None
        Random seed.

    Attributes
    ----------
    estimators_ : list[DecisionTreeClassifier]
        Fitted decision trees.
    classes_ : np.ndarray
        Unique class labels.
    feature_importances_ : np.ndarray
        Mean feature importance across all trees.
    n_features_in_ : int
        Number of features seen during fitting.
    """

    def __init__(
        self,
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        criterion="gini",
        max_features="sqrt",
        bootstrap=True,
        random_state=None,
    ):
        if n_estimators <= 0:
            raise ValueError("n_estimators must be positive.")

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state

        self.estimators_ = None
        self.classes_ = None
        self.feature_importances_ = None
        self.n_features_in_ = None
        self.bootstrap_indices_ = None

    def fit(self, X, y):
        """
        Fit the random forest classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Class labels.

        Returns
        -------
        RandomForestClassifier
            Fitted forest.
        """
        X, y = _validate_X_y(X, y)

        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        self.estimators_ = []
        self.bootstrap_indices_ = []

        rng = np.random.default_rng(self.random_state)

        for _ in range(self.n_estimators):
            if self.bootstrap:
                sample_indices = rng.integers(0, X.shape[0], size=X.shape[0])
            else:
                sample_indices = np.arange(X.shape[0])

            tree_seed = int(rng.integers(0, np.iinfo(np.int32).max))

            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                criterion=self.criterion,
                max_features=self.max_features,
                random_state=tree_seed,
            )

            tree.fit(X[sample_indices], y[sample_indices])

            self.estimators_.append(tree)
            self.bootstrap_indices_.append(sample_indices)

        self.feature_importances_ = self._compute_feature_importances()

        return self

    def predict(self, X):
        """
        Predict class labels for input samples by majority vote.

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

        tree_predictions = np.asarray([tree.predict(X) for tree in self.estimators_]).T

        return np.array([self._majority_vote(row) for row in tree_predictions])

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

    @staticmethod
    def _majority_vote(predictions):
        """
        Return the most common label among tree predictions.

        Parameters
        ----------
        predictions : array-like
            Predicted labels from all trees for one sample.

        Returns
        -------
        object
            Majority-vote class label.
        """
        counts = Counter(predictions)
        max_count = max(counts.values())
        tied_labels = [label for label, count in counts.items() if count == max_count]

        return sorted(tied_labels)[0]

    def _compute_feature_importances(self):
        """
        Average feature importances across trees.

        Returns
        -------
        np.ndarray
            Normalized feature importance scores.
        """
        importances = np.array(
            [tree.feature_importances_ for tree in self.estimators_],
            dtype=float,
        )

        mean_importances = np.mean(importances, axis=0)
        total = np.sum(mean_importances)

        if total > 0:
            return mean_importances / total

        return mean_importances

    def _check_is_fitted(self):
        if self.estimators_ is None:
            raise ValueError("This RandomForestClassifier instance is not fitted yet.")

    def _check_n_features(self, X):
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, but got {X.shape[1]}."
            )


class RandomForestRegressor:
    """
    Random forest regressor implemented from scratch.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of decision trees in the forest.
    max_depth : int or None, default=None
        Maximum depth of each tree.
    min_samples_split : int, default=2
        Minimum number of samples required to split an internal node.
    min_samples_leaf : int, default=1
        Minimum number of samples required at each leaf node.
    criterion : {"squared_error", "absolute_error"}, default="squared_error"
        Impurity criterion used by each decision tree.
    max_features : int, {"sqrt", "log2"}, or None, default=None
        Number of features considered at each split.
    bootstrap : bool, default=True
        Whether to train each tree on a bootstrap sample.
    random_state : int or None, default=None
        Random seed.

    Attributes
    ----------
    estimators_ : list[DecisionTreeRegressor]
        Fitted decision trees.
    feature_importances_ : np.ndarray
        Mean feature importance across all trees.
    n_features_in_ : int
        Number of features seen during fitting.
    """

    def __init__(
        self,
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        criterion="squared_error",
        max_features=None,
        bootstrap=True,
        random_state=None,
    ):
        if n_estimators <= 0:
            raise ValueError("n_estimators must be positive.")

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state

        self.estimators_ = None
        self.feature_importances_ = None
        self.n_features_in_ = None
        self.bootstrap_indices_ = None

    def fit(self, X, y):
        """
        Fit the random forest regressor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        RandomForestRegressor
            Fitted forest.
        """
        X, y = _validate_X_y(X, y)
        y = y.astype(float)

        self.n_features_in_ = X.shape[1]
        self.estimators_ = []
        self.bootstrap_indices_ = []

        rng = np.random.default_rng(self.random_state)

        for _ in range(self.n_estimators):
            if self.bootstrap:
                sample_indices = rng.integers(0, X.shape[0], size=X.shape[0])
            else:
                sample_indices = np.arange(X.shape[0])

            tree_seed = int(rng.integers(0, np.iinfo(np.int32).max))

            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                criterion=self.criterion,
                max_features=self.max_features,
                random_state=tree_seed,
            )

            tree.fit(X[sample_indices], y[sample_indices])

            self.estimators_.append(tree)
            self.bootstrap_indices_.append(sample_indices)

        self.feature_importances_ = self._compute_feature_importances()

        return self

    def predict(self, X):
        """
        Predict target values by averaging predictions across trees.

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

        tree_predictions = np.asarray([tree.predict(X) for tree in self.estimators_])

        return np.mean(tree_predictions, axis=0)

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

    def _compute_feature_importances(self):
        """
        Average feature importances across trees.

        Returns
        -------
        np.ndarray
            Normalized feature importance scores.
        """
        importances = np.array(
            [tree.feature_importances_ for tree in self.estimators_],
            dtype=float,
        )

        mean_importances = np.mean(importances, axis=0)
        total = np.sum(mean_importances)

        if total > 0:
            return mean_importances / total

        return mean_importances

    def _check_is_fitted(self):
        if self.estimators_ is None:
            raise ValueError("This RandomForestRegressor instance is not fitted yet.")

    def _check_n_features(self, X):
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, but got {X.shape[1]}."
            )