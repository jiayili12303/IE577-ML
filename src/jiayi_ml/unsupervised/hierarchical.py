"""
Agglomerative hierarchical clustering implemented from scratch using NumPy.

This module provides an AgglomerativeClustering class with a scikit-learn-like
API. The algorithm starts with each observation as its own cluster and repeatedly
merges the closest pair of clusters until the desired number of clusters remains.
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


class AgglomerativeClustering:
    """
    Agglomerative hierarchical clustering implemented from scratch.

    Parameters
    ----------
    n_clusters : int, default=2
        Number of clusters to form.
    linkage : {"single", "complete", "average"}, default="average"
        Linkage criterion used to compute distance between clusters.
        - "single": minimum distance between points in two clusters.
        - "complete": maximum distance between points in two clusters.
        - "average": average distance between points in two clusters.
    metric : {"euclidean", "manhattan"}, default="euclidean"
        Distance metric used between individual samples.

    Attributes
    ----------
    labels_ : np.ndarray
        Cluster labels for each training sample.
    children_ : np.ndarray
        Pairs of clusters merged at each step. Original samples are indexed
        from 0 to n_samples - 1. Newly formed clusters are indexed starting
        from n_samples.
    distances_ : np.ndarray
        Distance between clusters at each merge step.
    n_features_in_ : int
        Number of features seen during fitting.
    """

    def __init__(
        self,
        n_clusters=2,
        linkage="average",
        metric="euclidean",
    ):
        if n_clusters <= 0:
            raise ValueError("n_clusters must be positive.")

        if linkage not in {"single", "complete", "average"}:
            raise ValueError("linkage must be one of 'single', 'complete', or 'average'.")

        if metric not in {"euclidean", "manhattan"}:
            raise ValueError("metric must be either 'euclidean' or 'manhattan'.")

        self.n_clusters = n_clusters
        self.linkage = linkage
        self.metric = metric

        self.labels_ = None
        self.children_ = None
        self.distances_ = None
        self.n_features_in_ = None

    def fit(self, X):
        """
        Fit agglomerative clustering on the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        AgglomerativeClustering
            Fitted clustering model.
        """
        X = _validate_X(X)
        n_samples, n_features = X.shape

        if self.n_clusters > n_samples:
            raise ValueError("n_clusters cannot be larger than number of samples.")

        self.n_features_in_ = n_features

        pairwise_distances = self._pairwise_distances(X)

        clusters = {i: [i] for i in range(n_samples)}
        active_cluster_ids = list(range(n_samples))
        next_cluster_id = n_samples

        children = []
        merge_distances = []

        while len(active_cluster_ids) > self.n_clusters:
            best_pair = None
            best_distance = np.inf

            for i in range(len(active_cluster_ids)):
                for j in range(i + 1, len(active_cluster_ids)):
                    cluster_id_a = active_cluster_ids[i]
                    cluster_id_b = active_cluster_ids[j]

                    distance = self._cluster_distance(
                        clusters[cluster_id_a],
                        clusters[cluster_id_b],
                        pairwise_distances,
                    )

                    if distance < best_distance:
                        best_distance = distance
                        best_pair = (cluster_id_a, cluster_id_b)

            cluster_id_a, cluster_id_b = best_pair

            children.append([cluster_id_a, cluster_id_b])
            merge_distances.append(best_distance)

            merged_cluster = clusters[cluster_id_a] + clusters[cluster_id_b]
            clusters[next_cluster_id] = merged_cluster

            active_cluster_ids.remove(cluster_id_a)
            active_cluster_ids.remove(cluster_id_b)
            active_cluster_ids.append(next_cluster_id)

            next_cluster_id += 1

        labels = np.empty(n_samples, dtype=int)

        sorted_final_clusters = sorted(
            active_cluster_ids,
            key=lambda cluster_id: min(clusters[cluster_id]),
        )

        for label, cluster_id in enumerate(sorted_final_clusters):
            for sample_idx in clusters[cluster_id]:
                labels[sample_idx] = label

        self.labels_ = labels
        self.children_ = np.asarray(children, dtype=int)
        self.distances_ = np.asarray(merge_distances, dtype=float)

        return self

    def fit_predict(self, X):
        """
        Fit the model and return cluster labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        np.ndarray
            Cluster labels.
        """
        return self.fit(X).labels_

    def _pairwise_distances(self, X):
        """
        Compute pairwise distances between all samples.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.

        Returns
        -------
        np.ndarray
            Pairwise distance matrix of shape (n_samples, n_samples).
        """
        if self.metric == "euclidean":
            diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
            return np.sqrt(np.sum(diff ** 2, axis=2))

        if self.metric == "manhattan":
            diff = np.abs(X[:, np.newaxis, :] - X[np.newaxis, :, :])
            return np.sum(diff, axis=2)

        raise ValueError("metric must be either 'euclidean' or 'manhattan'.")

    def _cluster_distance(self, cluster_a, cluster_b, pairwise_distances):
        """
        Compute distance between two clusters using the selected linkage rule.

        Parameters
        ----------
        cluster_a : list[int]
            Sample indices in the first cluster.
        cluster_b : list[int]
            Sample indices in the second cluster.
        pairwise_distances : np.ndarray
            Pairwise sample distance matrix.

        Returns
        -------
        float
            Distance between the two clusters.
        """
        distances = pairwise_distances[np.ix_(cluster_a, cluster_b)]

        if self.linkage == "single":
            return float(np.min(distances))

        if self.linkage == "complete":
            return float(np.max(distances))

        if self.linkage == "average":
            return float(np.mean(distances))

        raise ValueError("linkage must be one of 'single', 'complete', or 'average'.")