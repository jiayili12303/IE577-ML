"""
K-Means clustering implemented from scratch using NumPy.
"""

from __future__ import annotations

import numpy as np


def _validate_X(X):
    X = np.asarray(X, dtype=float)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    if X.ndim != 2:
        raise ValueError("X must be a 1D or 2D array.")

    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X must not be empty.")

    return X


class KMeans:
    """
    K-Means clustering implemented from scratch.

    Parameters
    ----------
    n_clusters : int, default=8
        Number of clusters.
    max_iter : int, default=300
        Maximum number of iterations.
    tol : float, default=1e-4
        Tolerance for convergence.
    n_init : int, default=10
        Number of random initializations.
    init : {"random", "k-means++"}, default="k-means++"
        Initialization method.
    random_state : int, optional
        Random seed.
    """

    def __init__(
        self,
        n_clusters=8,
        max_iter=300,
        tol=1e-4,
        n_init=10,
        init="k-means++",
        random_state=None,
    ):
        if n_clusters <= 0:
            raise ValueError("n_clusters must be positive.")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive.")
        if tol < 0:
            raise ValueError("tol must be nonnegative.")
        if n_init <= 0:
            raise ValueError("n_init must be positive.")
        if init not in {"random", "k-means++"}:
            raise ValueError("init must be either 'random' or 'k-means++'.")

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.init = init
        self.random_state = random_state

        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = None
        self.n_features_in_ = None

    def fit(self, X):
        X = _validate_X(X)

        if self.n_clusters > X.shape[0]:
            raise ValueError("n_clusters cannot be larger than number of samples.")

        self.n_features_in_ = X.shape[1]
        rng = np.random.default_rng(self.random_state)

        best_centers = None
        best_labels = None
        best_inertia = np.inf
        best_n_iter = 0

        for _ in range(self.n_init):
            centers = self._initialize_centers(X, rng)

            for iteration in range(1, self.max_iter + 1):
                labels = self._assign_labels(X, centers)
                new_centers = self._update_centers(X, labels, centers)

                center_shift = np.linalg.norm(new_centers - centers)
                centers = new_centers

                if center_shift <= self.tol:
                    break

            labels = self._assign_labels(X, centers)
            inertia = self._compute_inertia(X, centers, labels)

            if inertia < best_inertia:
                best_centers = centers.copy()
                best_labels = labels.copy()
                best_inertia = inertia
                best_n_iter = iteration

        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = float(best_inertia)
        self.n_iter_ = best_n_iter

        return self

    def predict(self, X):
        self._check_is_fitted()
        X = _validate_X(X)
        self._check_n_features(X)

        return self._assign_labels(X, self.cluster_centers_)

    def fit_predict(self, X):
        return self.fit(X).labels_

    def transform(self, X):
        self._check_is_fitted()
        X = _validate_X(X)
        self._check_n_features(X)

        return np.sqrt(self._pairwise_squared_distances(X, self.cluster_centers_))

    def _initialize_centers(self, X, rng):
        if self.init == "random":
            indices = rng.choice(X.shape[0], size=self.n_clusters, replace=False)
            return X[indices].copy()

        centers = np.empty((self.n_clusters, X.shape[1]), dtype=float)
        first_index = rng.integers(X.shape[0])
        centers[0] = X[first_index]

        closest_distances = self._pairwise_squared_distances(X, centers[[0]]).ravel()

        for center_idx in range(1, self.n_clusters):
            total_distance = np.sum(closest_distances)

            if total_distance == 0:
                indices = rng.choice(X.shape[0], size=self.n_clusters, replace=False)
                return X[indices].copy()

            probabilities = closest_distances / total_distance
            next_index = rng.choice(X.shape[0], p=probabilities)
            centers[center_idx] = X[next_index]

            new_distances = self._pairwise_squared_distances(
                X, centers[[center_idx]]
            ).ravel()
            closest_distances = np.minimum(closest_distances, new_distances)

        return centers

    def _assign_labels(self, X, centers):
        distances = self._pairwise_squared_distances(X, centers)
        return np.argmin(distances, axis=1)

    def _update_centers(self, X, labels, old_centers):
        new_centers = old_centers.copy()

        for cluster_idx in range(self.n_clusters):
            cluster_points = X[labels == cluster_idx]

            if cluster_points.shape[0] > 0:
                new_centers[cluster_idx] = np.mean(cluster_points, axis=0)
            else:
                distances = self._pairwise_squared_distances(X, old_centers)
                closest_distances = np.min(distances, axis=1)
                farthest_index = int(np.argmax(closest_distances))
                new_centers[cluster_idx] = X[farthest_index]

        return new_centers

    @staticmethod
    def _pairwise_squared_distances(X, centers):
        return np.sum((X[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2, axis=2)

    def _compute_inertia(self, X, centers, labels):
        distances = self._pairwise_squared_distances(X, centers)
        return float(np.sum(distances[np.arange(X.shape[0]), labels]))

    def _check_is_fitted(self):
        if self.cluster_centers_ is None or self.labels_ is None:
            raise ValueError("This KMeans instance is not fitted yet.")

    def _check_n_features(self, X):
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, but got {X.shape[1]}."
            )
