"""
DBSCAN clustering implemented from scratch using NumPy.

DBSCAN stands for Density-Based Spatial Clustering of Applications with Noise.
It groups together points that are closely packed and labels points in low-density
regions as noise.
"""

from __future__ import annotations

import numpy as np


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


class DBSCAN:
    """
    Density-based spatial clustering implemented from scratch.

    Parameters
    ----------
    eps : float, default=0.5
        Maximum distance between two samples for one to be considered in the
        neighborhood of the other.
    min_samples : int, default=5
        Minimum number of samples in a neighborhood for a point to be considered
        a core point.
    metric : {"euclidean", "manhattan"}, default="euclidean"
        Distance metric used to define neighborhoods.

    Attributes
    ----------
    labels_ : np.ndarray
        Cluster labels for each training sample. Noise points are labeled -1.
    core_sample_indices_ : np.ndarray
        Indices of core samples.
    components_ : np.ndarray
        Core samples.
    n_features_in_ : int
        Number of features seen during fitting.
    """

    def __init__(self, eps=0.5, min_samples=5, metric="euclidean"):
        if eps <= 0:
            raise ValueError("eps must be positive.")

        if min_samples <= 0:
            raise ValueError("min_samples must be positive.")

        if metric not in {"euclidean", "manhattan"}:
            raise ValueError("metric must be either 'euclidean' or 'manhattan'.")

        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric

        self.labels_ = None
        self.core_sample_indices_ = None
        self.components_ = None
        self.n_features_in_ = None

    def fit(self, X):
        """
        Fit DBSCAN clustering on the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        DBSCAN
            Fitted clustering model.
        """
        X = _validate_X(X)
        n_samples = X.shape[0]

        self.n_features_in_ = X.shape[1]

        labels = np.full(n_samples, -1, dtype=int)
        visited = np.zeros(n_samples, dtype=bool)
        core_mask = np.zeros(n_samples, dtype=bool)

        neighborhoods = [self._region_query(X, i) for i in range(n_samples)]

        for i, neighbors in enumerate(neighborhoods):
            if len(neighbors) >= self.min_samples:
                core_mask[i] = True

        cluster_id = 0

        for point_idx in range(n_samples):
            if visited[point_idx]:
                continue

            visited[point_idx] = True
            neighbors = neighborhoods[point_idx]

            if len(neighbors) < self.min_samples:
                labels[point_idx] = -1
            else:
                self._expand_cluster(
                    labels=labels,
                    visited=visited,
                    neighborhoods=neighborhoods,
                    point_idx=point_idx,
                    neighbors=neighbors,
                    cluster_id=cluster_id,
                )
                cluster_id += 1

        self.labels_ = labels
        self.core_sample_indices_ = np.where(core_mask)[0]
        self.components_ = X[self.core_sample_indices_]

        return self

    def fit_predict(self, X):
        """
        Fit DBSCAN and return cluster labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        np.ndarray
            Cluster labels. Noise points are labeled -1.
        """
        return self.fit(X).labels_

    def _expand_cluster(
        self,
        labels,
        visited,
        neighborhoods,
        point_idx,
        neighbors,
        cluster_id,
    ):
        """
        Expand one DBSCAN cluster from a core point.

        Parameters
        ----------
        labels : np.ndarray
            Current labels.
        visited : np.ndarray
            Boolean array indicating whether each point has been visited.
        neighborhoods : list[np.ndarray]
            Precomputed neighborhood indices.
        point_idx : int
            Starting core point index.
        neighbors : np.ndarray
            Neighbor indices of the starting point.
        cluster_id : int
            Cluster label to assign.
        """
        labels[point_idx] = cluster_id
        seeds = list(neighbors)

        seed_position = 0
        while seed_position < len(seeds):
            neighbor_idx = seeds[seed_position]

            if not visited[neighbor_idx]:
                visited[neighbor_idx] = True
                neighbor_neighbors = neighborhoods[neighbor_idx]

                if len(neighbor_neighbors) >= self.min_samples:
                    for new_neighbor in neighbor_neighbors:
                        if new_neighbor not in seeds:
                            seeds.append(int(new_neighbor))

            if labels[neighbor_idx] == -1:
                labels[neighbor_idx] = cluster_id

            seed_position += 1

    def _region_query(self, X, point_idx):
        """
        Find all samples within eps of a given point.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        point_idx : int
            Index of query point.

        Returns
        -------
        np.ndarray
            Indices of points within eps distance, including the query point.
        """
        distances = self._compute_distances(X, X[point_idx])
        return np.where(distances <= self.eps)[0]

    def _compute_distances(self, X, x):
        """
        Compute distances from all samples to one query point.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        x : np.ndarray
            Query point.

        Returns
        -------
        np.ndarray
            Distances to query point.
        """
        if self.metric == "euclidean":
            return np.sqrt(np.sum((X - x) ** 2, axis=1))

        if self.metric == "manhattan":
            return np.sum(np.abs(X - x), axis=1)

        raise ValueError("metric must be either 'euclidean' or 'manhattan'.")
